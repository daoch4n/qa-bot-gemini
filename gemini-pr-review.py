import json
import os
import sys
import time
import random
import datetime
import urllib.parse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Iterable
import google.generativeai as Client
from github import Github, GithubException
import requests
import fnmatch
import re
from unidiff import Hunk, PatchedFile, PatchSet
from unidiff.patch import Line

# Initialize clients
def initialize_clients():
    try:
        if os.environ.get("GEMINI_TEST_MODE") == "1":
            print("Test mode: Skipping GitHub and Gemini client initialization")
            return None, None

        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            print("Error: GITHUB_TOKEN environment variable is required.")
            sys.exit(1)
        gh_client = Github(github_token)

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Error: GEMINI_API_KEY environment variable is required.")
            sys.exit(1)
        Client.configure(api_key=gemini_api_key)
        gemini_client_module = Client # Use the configured module

        return gh_client, gemini_client_module
    except Exception as e:
        print(f"Error during client initialization: {e}")
        traceback.print_exc()
        sys.exit(1)

gh, gemini_client_module = initialize_clients()


class PRDetails:
    def __init__(self, owner: str, repo_name_str: str, pull_number: int, title: str, description: str, repo_obj=None, pr_obj=None, event_type: str = None):
        self.owner = owner
        self.repo_name = repo_name_str
        self.pull_number = pull_number
        self.title = title
        self.description = description
        self.repo_obj = repo_obj
        self.pr_obj = pr_obj
        self.event_type = event_type

    def get_full_repo_name(self):
        return f"{self.owner}/{self.repo_name}"


def get_pr_details() -> PRDetails:
    github_event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not github_event_path:
        print("Error: GITHUB_EVENT_PATH environment variable not set.")
        sys.exit(1)

    with open(github_event_path, "r", encoding="utf-8") as f:
        event_data = json.load(f)

    event_name = os.environ.get("GITHUB_EVENT_NAME")
    pr_event_type = None

    if event_name == "issue_comment":
        if "issue" in event_data and "pull_request" in event_data["issue"]:
            pull_number = event_data["issue"]["number"]
            repo_full_name = event_data["repository"]["full_name"]
            pr_event_type = "comment"
        else:
            print("Error: issue_comment event not on a pull request.")
            sys.exit(1)
    elif event_name == "pull_request":
        pull_number = event_data["pull_request"]["number"]
        repo_full_name = event_data["repository"]["full_name"]
        pr_event_type = event_data.get("action")
        print(f"Pull request event action: {pr_event_type}")
    else:
        print(f"Error: Unsupported GITHUB_EVENT_NAME: {event_name}")
        sys.exit(1)

    owner, repo_name_str = repo_full_name.split("/")

    try:
        repo_obj = gh.get_repo(repo_full_name)
        pr_obj = repo_obj.get_pull(pull_number)
    except GithubException as e:
        print(f"Error accessing GitHub repository or PR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while fetching PR details: {e}")
        sys.exit(1)

    return PRDetails(owner, repo_name_str, pull_number, pr_obj.title, pr_obj.body or "", repo_obj, pr_obj, pr_event_type)


def get_diff(pr_details: PRDetails, comparison_sha: Optional[str] = None) -> str:
    repo = pr_details.repo_obj
    pr = pr_details.pr_obj
    head_sha = pr.head.sha

    if comparison_sha:
        print(f"Getting diff comparing HEAD ({head_sha}) against specified SHA ({comparison_sha})")
        try:
            comparison_obj = repo.compare(comparison_sha, head_sha)
            diff_parts = []
            for file_diff in comparison_obj.files:
                if file_diff.patch:
                    # Construct a valid diff header format for unidiff
                    source_file_path_for_header = file_diff.previous_filename if file_diff.status == 'renamed' else file_diff.filename
                    target_file_path_for_header = file_diff.filename

                    diff_header = f"diff --git a/{source_file_path_for_header} b/{target_file_path_for_header}\n"
                    if file_diff.status == 'added':
                        diff_header += f"new file mode {getattr(file_diff, 'mode', '100644')}\n"
                        diff_header += f"index 0000000..{file_diff.sha[:7]}\n"
                    elif file_diff.status == 'deleted':
                        diff_header += f"deleted file mode {getattr(file_diff, 'mode', '100644')}\n"
                        diff_header += f"index {file_diff.sha[:7]}..0000000\n"
                    elif file_diff.status == 'renamed':
                        diff_header += f"similarity index {getattr(file_diff, 'similarity_index', '100')}%\n"
                        diff_header += f"rename from {source_file_path_for_header}\n" # already set as prev_filename
                        diff_header += f"rename to {target_file_path_for_header}\n"   # already set as filename
                        if hasattr(file_diff, 'sha'): # If it's a rename with modifications
                             diff_header += f"index {getattr(file_diff, 'previous_sha', '0000000')[:7]}..{file_diff.sha[:7]}\n"
                    elif file_diff.status == 'modified':
                         # For modified files, the index line shows old SHA..new SHA
                         # PyGithub's file_diff.sha is the new SHA. We need the old one if available,
                         # or rely on the patch content itself to have it.
                         # For simplicity, we'll rely on the patch content for modified index line.
                         pass


                    patch_content = file_diff.patch

                    # Ensure --- and +++ lines are present, this is critical for unidiff
                    # The patch from GitHub API usually has these, but repo.compare() might be different.
                    lines = patch_content.splitlines()
                    final_patch_lines = []

                    # Check if patch already contains valid ---/+++ for THESE filenames
                    # This logic can be complex if file_diff.patch is not a standard unidiff snippet
                    # For repo.compare, file_diff.patch should be a standard diff hunk content.

                    # Simplification: Assume file_diff.patch from repo.compare is the core hunk data
                    # and we need to wrap it correctly for unidiff.
                    final_patch_lines.append(f"--- a/{source_file_path_for_header}")
                    final_patch_lines.append(f"+++ b/{target_file_path_for_header}")
                    final_patch_lines.extend(lines) # Add the actual patch lines (hunks)

                    diff_parts.append(diff_header + "\n".join(final_patch_lines))

            if diff_parts:
                diff_text = "\n".join(diff_parts) # Each element in diff_parts is a full diff for one file
                print(f"Retrieved diff (length: {len(diff_text)}) using repo.compare('{comparison_sha}', '{head_sha}')")
                return diff_text
            else:
                print(f"No changes found comparing {comparison_sha} to {head_sha}")
                return ""
        except GithubException as e:
            print(f"Error getting comparison diff (compare {comparison_sha} vs {head_sha}): {e}. Falling back.")
        except Exception as e:
            print(f"Unexpected error during repo.compare: {e}. Falling back.")
            traceback.print_exc()


    print(f"Falling back to pr.get_diff() for PR #{pr_details.pull_number}")
    try:
        diff_text = pr.get_diff() # This is usually well-formatted for unidiff
        if diff_text:
            print(f"Retrieved diff (length: {len(diff_text)}) using pr.get_diff()")
            return diff_text
        else:
            print("pr.get_diff() returned no content.")
            return ""
    except GithubException as e:
        print(f"Error getting diff using pr.get_diff(): {e}. Falling back further.")
    except Exception as e:
        print(f"Unexpected error during pr.get_diff(): {e}. Falling back further.")

    print(f"Falling back to direct API request for PR diff for PR #{pr_details.pull_number}")
    api_url = f"https://api.github.com/repos/{pr_details.get_full_repo_name()}/pulls/{pr_details.pull_number}"
    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}',
        'Accept': 'application/vnd.github.v3.diff'
    }
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        diff_text = response.text
        print(f"Retrieved diff (length: {len(diff_text)}) via direct API call.")
        return diff_text
    except requests.exceptions.RequestException as e:
        print(f"Failed to get diff via direct API call: {e}")
    except Exception as e:
        print(f"Unexpected error during direct API call for diff: {e}")

    print("All methods to retrieve diff failed.")
    return ""


def get_hunk_representation(hunk: Hunk) -> str:
    return str(hunk)


def get_file_content(file_path: str) -> str:
    full_file_content = ""
    code_extensions = [
        ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss", ".java",
        ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".php", ".rb", ".sh", ".bash",
        ".json", ".yml", ".yaml", ".toml", ".md"
    ]
    is_code_file = any(file_path.endswith(ext) for ext in code_extensions)

    if not is_code_file:
        print(f"Skipping full file context for non-code or binary-like file: {file_path}")
        return ""

    try:
        p_file_path = Path(file_path)
        if p_file_path.exists() and p_file_path.is_file():
            file_stat = p_file_path.stat()
            max_initial_read_bytes = 300000

            if file_stat.st_size > max_initial_read_bytes:
                print(f"File {file_path} is very large ({file_stat.st_size} bytes). Reading a truncated version for context.")
                with open(p_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    start_content = f.read(max_initial_read_bytes // 2)
                full_file_content = start_content + "\n\n... [content truncated due to very large size] ...\n\n"
            else:
                 with open(p_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_file_content = f.read()

            max_char_len_for_context = 150000
            if len(full_file_content) > max_char_len_for_context:
                print(f"File content for {file_path} still too long after initial read ({len(full_file_content)} chars), further truncating for Gemini context.")
                half_len = max_char_len_for_context // 2
                full_file_content = full_file_content[:half_len] + \
                                    "\n\n... [content context truncated for brevity] ...\n\n" + \
                                    full_file_content[-half_len:]

            print(f"Read file content for {file_path} (length: {len(full_file_content)} chars after potential truncation).")
        else:
            print(f"File {file_path} does not exist locally or is not a file. Cannot provide full context.")
    except Exception as e:
        print(f"Error reading full file content for {file_path}: {e}")
        traceback.print_exc()
    return full_file_content


def load_previous_review_data(filepath_str: str = "reviews/gemini-pr-review.json") -> Dict[str, Any]:
    """Load previous review data from JSON file if it exists."""
    filepath = Path(filepath_str)
    if not filepath.exists():
        print(f"Previous review file {filepath_str} not found. No previous context will be provided.")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Successfully loaded previous review data from {filepath_str}")
            return data
    except Exception as e:
        print(f"Error loading previous review data from {filepath_str}: {e}")
        return {}


def get_previous_file_comments(review_data: Dict[str, Any], file_path: str) -> List[Dict[str, Any]]:
    """Extract previous comments for a specific file from the review data."""
    if not review_data or "review_comments" not in review_data:
        return []

    file_comments = []
    for comment in review_data.get("review_comments", []):
        if comment.get("file_path") == file_path:
            file_comments.append(comment)

    print(f"Found {len(file_comments)} previous comments for file {file_path}")
    return file_comments


def create_batch_prompt(patched_file: PatchedFile, pr_details: PRDetails) -> str:
    full_file_content_for_context = get_file_content(patched_file.path)

    # Load previous review data
    previous_review_data = load_previous_review_data()
    previous_file_comments = get_previous_file_comments(previous_review_data, patched_file.path)

    combined_hunks_text = ""
    for i, hunk in enumerate(patched_file):
        hunk_text = get_hunk_representation(hunk)
        if not hunk_text.strip():
            continue

        separator = ("-" * 20) + f" Hunk {i+1} (0-indexed: {i}) " + ("-" * 20) + "\n"
        combined_hunks_text += ("\n\n" if i > 0 else "") + separator + hunk_text

    instructions = """Your task is reviewing pull requests. Instructions:
- Provide the response in the following JSON format: {"reviews": [{"hunkIndex": <hunk_index_0_based>, "lineNumber": <line_number_in_hunk_content_1_based>, "reviewComment": "<review_comment_using_github_markdown>", "confidence": "<High|Medium|Low>"}]}
- `hunkIndex` is 0-based, referring to which hunk in the *provided diff below* the comment applies to (matches the 'Hunk X (0-indexed: Y)' header).
- `lineNumber` is 1-based, relative to the *content lines* within that specific hunk (i.e., line 1 is the first line *after* the '@@ ... @@' header of that hunk). These are the lines starting with '+', '-', or space.
- `confidence` indicates your certainty and the potential impact: "High" (likely critical issue), "Medium" (potential issue/best practice), "Low" (minor suggestion/nitpick).
- Provide comments if there is something genuinely to improve or discuss. If no issues, "reviews" should be an empty array. Consider the severity of the issue when deciding to comment.
- Use GitHub Markdown for `reviewComment`.
- Focus on: bugs, security vulnerabilities, performance bottlenecks, unclear logic, anti-patterns, and violations of SOLID principles or other key design patterns. High-impact issues are preferred.
- Make comments actionable. Suggest improvements or ask clarifying questions.
- DO NOT suggest adding comments to the code itself (e.g., "add a comment here explaining X").
- NOTE: Basic formatting/linting is handled by Biome. Focus on substantive issues. Do not comment on minor style issues. You are reviewing the *final* auto-formatted/linted code.
- Carefully analyze the full file context (if provided) and PR context before making suggestions to avoid hallucinations or irrelevant points.
- Only suggest changes relevant to the diff. Do not comment on unrelated code unless directly impacted by the changes in the diff.
- Be concise and clear.
- If previous review comments are provided, consider them in your analysis. Don't repeat the same issues if they haven't been addressed, but you can provide updated feedback if the code has changed in those areas.
"""

    pr_context = f"\nPull Request Title: {pr_details.title}\nPull Request Description:\n---\n{pr_details.description or 'No description provided.'}\n---\n"

    # Add previous review context if available
    previous_review_context = ""
    if previous_file_comments:
        previous_review_context = "\nPrevious Review Comments for this file:\n"
        for i, comment in enumerate(previous_file_comments):
            previous_review_context += f"Comment {i+1}:\n"
            previous_review_context += f"- File: {comment.get('file_path')}\n"
            previous_review_context += f"- Category: {comment.get('detected_category_heuristic', 'N/A')}\n"
            previous_review_context += f"- Severity: {comment.get('detected_severity_heuristic', 'N/A')}\n"
            previous_review_context += f"- Content: {comment.get('comment_text_md', 'N/A')}\n\n"

    file_context_header = ""
    file_content_block = ""
    if full_file_content_for_context:
        file_context_header = "\nFull content of the file for better context (it may be truncated if too large):\n"
        file_ext = Path(patched_file.path).suffix[1:]
        file_content_block = f"```{file_ext or 'text'}\n{full_file_content_for_context}\n```\n"

    diff_to_review_header = f"\nReview the following code diffs for the file \"{patched_file.path}\" ({len(list(patched_file))} hunks):\n"
    diff_block = f"```diff\n{combined_hunks_text}\n```"

    return instructions + pr_context + previous_review_context + file_context_header + file_content_block + diff_to_review_header + diff_block


LAST_GEMINI_REQUEST_TIME = 0
GEMINI_RPM_LIMIT = 45
GEMINI_REQUEST_INTERVAL_SECONDS = 60.0 / GEMINI_RPM_LIMIT

def enforce_gemini_rate_limits():
    global LAST_GEMINI_REQUEST_TIME
    current_time = time.time()
    time_since_last = current_time - LAST_GEMINI_REQUEST_TIME
    if time_since_last < GEMINI_REQUEST_INTERVAL_SECONDS:
        wait_time = GEMINI_REQUEST_INTERVAL_SECONDS - time_since_last
        print(f"Gemini Rate Limiter: Waiting {wait_time:.2f} seconds.")
        time.sleep(wait_time)
    LAST_GEMINI_REQUEST_TIME = time.time()


def get_ai_response_with_retry(prompt: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    model_name = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash-latest')

    if not gemini_client_module:
        print("Error: Gemini client module not initialized. Cannot make API call.")
        return []

    try:
        gemini_model = gemini_client_module.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error creating GenerativeModel instance with {model_name}: {e}")
        return []

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.5, # Increased slightly from 0.4
        "top_p": 0.95,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # Log the prompt before sending
    # For very long prompts, log only a summary or start/end
    prompt_log_max_len = 2000
    if len(prompt) > prompt_log_max_len:
        print(f"Full prompt (length {len(prompt)}). Start:\n{prompt[:prompt_log_max_len//2]}...\n...End:\n{prompt[-(prompt_log_max_len//2):]}")
    else:
        print(f"Full prompt:\n{prompt}")


    for attempt in range(1, max_retries + 1):
        try:
            enforce_gemini_rate_limits()
            print(f"Attempt {attempt}/{max_retries} - Sending prompt to Gemini model {model_name}...")

            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if not response.parts:
                print(f"Warning: AI response (attempt {attempt}) was empty or blocked. Prompt safety ratings: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    print(f"Prompt blocked due to: {response.prompt_feedback.block_reason_message}")
                if attempt < max_retries:
                    time.sleep( (2 ** attempt) * 2 )
                    continue
                return []


            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[len("```json"):]
            if response_text.endswith("```"):
                response_text = response_text[:-len("```")]
            response_text = response_text.strip()

            # Log the raw response text before parsing, for debugging "no suggestions"
            print(f"AI Response Text (attempt {attempt}, cleaned for JSON parsing):\n{response_text}")

            data = json.loads(response_text)

            if not isinstance(data, dict) or "reviews" not in data or not isinstance(data["reviews"], list):
                print(f"Error: AI response has invalid structure. Expected {{'reviews': [...]}}. Got: {type(data)}")
                if attempt < max_retries: time.sleep( (2 ** attempt) ); continue
                else: return []

            valid_reviews = []
            for i, review_item in enumerate(data["reviews"]):
                if not isinstance(review_item, dict):
                    print(f"Error: Review item {i} is not a dict: {review_item}")
                    continue
                required_keys = ["hunkIndex", "lineNumber", "reviewComment", "confidence"]
                if not all(k in review_item for k in required_keys):
                    print(f"Error: Review item {i} missing one or more required keys ({', '.join(required_keys)}): {review_item}")
                    continue
                try:
                    review_item["hunkIndex"] = int(review_item["hunkIndex"])
                    review_item["lineNumber"] = int(review_item["lineNumber"])
                except ValueError:
                    print(f"Error: Review item {i} hunkIndex or lineNumber not an int: {review_item}")
                    continue
                if review_item["confidence"] not in ["High", "Medium", "Low"]:
                    print(f"Warning: Review item {i} has invalid confidence '{review_item.get('confidence')}'. Defaulting to Low.")
                    review_item["confidence"] = "Low"

                valid_reviews.append(review_item)

            return valid_reviews

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from AI response (attempt {attempt}): {e}")
            response_text_for_log = "N/A"
            # 'response_text' is already defined from the try block
            if 'response_text' in locals() : response_text_for_log = response_text
            elif 'response' in locals() and hasattr(response, 'text'): response_text_for_log = response.text


            print(f"Response text that failed parsing (first 500 chars): '{response_text_for_log[:500]}'")
            if attempt == max_retries: return []
            time.sleep( (2 ** attempt) )
        except Exception as e:
            print(f"Error during Gemini API call (attempt {attempt}): {type(e).__name__} - {e}")
            if "rate limit" in str(e).lower() or "429" in str(e) or "ResourceExhausted" in type(e).__name__:
                delay = (2 ** attempt) + random.uniform(0,1)
                print(f"Rate limit likely hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            elif attempt == max_retries:
                 print(f"Max retries ({max_retries}) reached. Giving up on this API call.")
                 return []
            else:
                 time.sleep(5 * attempt)

    return []


def analyze_code(files_to_review: Iterable[PatchedFile], pr_details: PRDetails) -> List[Dict[str, Any]]:
    files_list = list(files_to_review)
    print(f"Starting code analysis for {len(files_list)} files.")
    all_comments_for_pr = []

    for patched_file in files_list:
        if not patched_file.path or patched_file.path == "/dev/null":
            print(f"Skipping file with invalid path: {patched_file.path}")
            continue

        hunks_in_file = list(patched_file)
        if not hunks_in_file:
            print(f"No hunks in file {patched_file.path}, skipping.")
            continue

        print(f"\nProcessing file: {patched_file.path} with {len(hunks_in_file)} hunks.")

        batch_prompt = create_batch_prompt(patched_file, pr_details)
        ai_reviews_for_file = get_ai_response_with_retry(batch_prompt)

        if ai_reviews_for_file:
            print(f"Received {len(ai_reviews_for_file)} review suggestions from AI for file {patched_file.path}.")
            file_comments = process_batch_ai_reviews(patched_file, ai_reviews_for_file)
            if file_comments:
                all_comments_for_pr.extend(file_comments)
        else:
            print(f"No review suggestions from AI for file {patched_file.path}.")

    print(f"\nFinished analysis. Total comments generated for PR: {len(all_comments_for_pr)}")
    return all_comments_for_pr


def get_hunk_header_str(hunk: Hunk) -> str:
    # A Hunk's string representation starts with its header: "@@ -old_start,old_len +new_start,new_len @@"
    # Or constructs it if not directly available.
    # For logging, it's useful.
    return f"@@ -{hunk.source_start},{hunk.source_length} +{hunk.target_start},{hunk.target_length} @@"


def calculate_github_position(file_patch: PatchedFile, target_hunk_obj: Hunk, relative_line_number_in_hunk_content: int) -> Optional[int]:
    cumulative_pos_in_diff = 0
    hunks_in_file = list(file_patch)

    target_hunk_found = False
    for current_hunk_obj in hunks_in_file:
        cumulative_pos_in_diff += 1

        if current_hunk_obj == target_hunk_obj:
            target_hunk_found = True
            comment_position = cumulative_pos_in_diff + relative_line_number_in_hunk_content -1

            num_content_lines_in_target_hunk = len(list(target_hunk_obj))
            if not (1 <= relative_line_number_in_hunk_content <= num_content_lines_in_target_hunk):
                target_hunk_header_str = get_hunk_header_str(target_hunk_obj) # Use helper
                print(f"Warning: AI suggested line {relative_line_number_in_hunk_content} which is outside the actual "
                      f"content lines ({num_content_lines_in_target_hunk}) of the target hunk in {file_patch.path}. "
                      f"Target Hunk Header: {target_hunk_header_str.strip()}. Skipping this comment.")
                return None
            return comment_position

        cumulative_pos_in_diff += len(list(current_hunk_obj))

    if not target_hunk_found:
        target_hunk_header_str = get_hunk_header_str(target_hunk_obj) # Use helper
        print(f"Error: Target hunk (header: {target_hunk_header_str.strip()}) not found by object comparison in file {file_patch.path} "
              f"during position calculation.")
    return None


def process_batch_ai_reviews(patched_file: PatchedFile, ai_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comments_for_github = []
    hunks_in_file = list(patched_file)

    for review_detail in ai_reviews:
        try:
            hunk_idx_from_ai = review_detail["hunkIndex"]
            line_num_in_hunk_content = review_detail["lineNumber"]
            comment_text = review_detail["reviewComment"]
            confidence = review_detail["confidence"]

            if not (0 <= hunk_idx_from_ai < len(hunks_in_file)):
                print(f"Warning: AI returned out-of-bounds hunkIndex {hunk_idx_from_ai} for file {patched_file.path} "
                      f"(has {len(hunks_in_file)} hunks). Skipping comment.")
                continue

            target_hunk_object = hunks_in_file[hunk_idx_from_ai]

            github_pos = calculate_github_position(patched_file, target_hunk_object, line_num_in_hunk_content)

            if github_pos is not None:
                formatted_comment_body = f"**AI Confidence: {confidence}**\n\n{comment_text}"

                gh_comment = {
                    "body": formatted_comment_body,
                    "path": patched_file.path,
                    "position": github_pos,
                    "confidence_raw": confidence
                }
                comments_for_github.append(gh_comment)
            else:
                print(f"Warning: Could not calculate GitHub position for comment in {patched_file.path}, "
                      f"Hunk Index {hunk_idx_from_ai}, Line {line_num_in_hunk_content}. Skipping.")

        except KeyError as e:
            print(f"Error processing AI review item due to missing key {e}: {review_detail}")
        except Exception as e:
            print(f"Unexpected error processing AI review item {review_detail}: {e}")
            traceback.print_exc()

    return comments_for_github


def save_review_results_to_json(pr_details: PRDetails, comments: List[Dict[str, Any]], filepath_str: str = "reviews/gemini-pr-review.json") -> str:
    filepath = Path(filepath_str)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    review_data = {
        "metadata": {
            "pr_number": pr_details.pull_number,
            "repo": pr_details.get_full_repo_name(),
            "title": pr_details.title,
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "review_tool": "Gemini AI Reviewer",
            "model_used": os.environ.get('GEMINI_MODEL', 'N/A')
        },
        "review_comments": []
    }

    for gh_comment_dict in comments:
        structured_comment = {
            "file_path": gh_comment_dict["path"],
            "github_diff_position": gh_comment_dict["position"],
            "comment_text_md": gh_comment_dict["body"],
            "ai_confidence": gh_comment_dict.get("confidence_raw", "N/A"),
            "detected_severity_heuristic": detect_severity(gh_comment_dict["body"]),
            "detected_category_heuristic": detect_category(gh_comment_dict["body"])
        }
        review_data["review_comments"].append(structured_comment)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    print(f"Review results saved to {filepath}")
    return str(filepath)


def detect_severity(comment_text: str) -> str:
    lower_text = comment_text.lower()
    if any(word in lower_text for word in ["critical", "security vulnerability", "crash", "exploit", "must fix", "data loss"]):
        return "critical"
    if any(word in lower_text for word in ["bug", "error", "incorrect", "wrong", "security", "potential vulnerability", "flaw"]):
        return "high"
    if any(word in lower_text for word in ["performance", "optimization", "memory", "leak", "consider fixing", "confusing", "unclear"]):
        return "medium"
    return "low"

def detect_category(comment_text: str) -> str:
    lower_text = comment_text.lower()
    if any(word in lower_text for word in ["security", "vulnerability", "exploit", "auth", "csrf", "xss", "injection", "password", "secret"]):
        return "security"
    if any(word in lower_text for word in ["performance", "slow", "optimization", "efficient", "memory", "cpu", "latency", "resource"]):
        return "performance"
    if any(word in lower_text for word in ["bug", "error", "incorrect", "wrong", "fix", "defect", "exception", "nullpointer"]):
        return "bug"
    if any(word in lower_text for word in ["style", "format", "naming", "convention", "readability", "clarity", "understandability", "documentation", "commenting"]):
        return "style/clarity"
    if any(word in lower_text for word in ["refactor", "clean", "simplify", "maintainability", "design", "architecture", "pattern", "anti-pattern", "duplication"]):
        return "refactoring/design"
    if any(word in lower_text for word in ["test", "coverage", "assertion", "mocking"]):
        return "testing"
    return "general"


def create_review_and_summary_comment(pr_details: PRDetails, comments_for_gh_review: List[Dict[str, Any]], review_json_path: str):
    if not pr_details.pr_obj:
        print("Error: PR object not available in PRDetails. Cannot create review or comments.")
        return

    pr = pr_details.pr_obj
    num_suggestions = len(comments_for_gh_review)

    if num_suggestions > 0:
        valid_review_comments = []
        for c in comments_for_gh_review:
            if all(k in c for k in ["body", "path", "position"]):
                if isinstance(c["position"], int) and isinstance(c["path"], str) and isinstance(c["body"], str):
                    valid_review_comments.append({
                        "body": c["body"],
                        "path": c["path"],
                        "position": c["position"]
                    })
                else:
                    print(f"Warning: Skipping malformed comment due to type mismatch: {c}")
            else:
                print(f"Warning: Skipping malformed comment due to missing keys: {c}")

        if valid_review_comments:
            try:
                print(f"Creating a PR review with {len(valid_review_comments)} suggestions.")
                pr.create_review(
                    body="Automated AI code review suggestions:",
                    event="COMMENT",
                    comments=valid_review_comments
                )
                print("Successfully created PR review with suggestions.")
            except GithubException as e:
                print(f"Error creating PR review: {e}. Status: {e.status}, Data: {e.data}")
                print("Falling back to posting individual issue comments for suggestions.")
                for c_item in valid_review_comments:
                    try:
                        pr.create_issue_comment(f"**File:** `{c_item['path']}` (at diff position {c_item['position']})\n\n{c_item['body']}")
                    except Exception as ie:
                        print(f"Error posting individual suggestion as issue comment: {ie}")
            except Exception as e:
                print(f"Unexpected error during PR review creation: {e}")
                traceback.print_exc()
        else:
            print("No validly structured comments to create a review with.")
    else:
        print("No suggestions to create a PR review for.")

    repo_full_name = os.environ.get("GITHUB_REPOSITORY", pr_details.get_full_repo_name())
    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    branch_name = os.environ.get("GITHUB_HEAD_REF")
    if not branch_name and hasattr(pr.head, 'ref'):
        branch_name = pr.head.ref

    review_file_url_md = f"Review JSON file (`{review_json_path}` in the repository)"
    if branch_name:
        try:
            encoded_branch = urllib.parse.quote_plus(branch_name)
            review_file_url = f"{server_url}/{repo_full_name}/blob/{encoded_branch}/{review_json_path}"
            review_file_url_md = f"Full review details in [`{review_json_path}`]({review_file_url})"
            print(f"Summary comment will link to: {review_file_url}")
        except Exception as url_e:
            print(f"Error creating review file URL: {url_e}")
    else:
        print("Warning: Could not determine branch name for summary comment URL.")


    summary_body = f"✨ **Gemini AI Code Review Complete** ✨\n\n"
    if num_suggestions > 0:
        summary_body += f"- Found {num_suggestions} potential areas for discussion/improvement (see review comments above or in the review tab).\n"
        summary_body += f"- {review_file_url_md}.\n"
    else:
        summary_body += "- No specific suggestions made by the AI in this pass.\n"
    summary_body += f"- Model: `{os.environ.get('GEMINI_MODEL', 'N/A')}`\n"

    # Add workflow run log link if available
    run_id = os.environ.get('GITHUB_RUN_ID')
    repo_name = os.environ.get('GITHUB_REPOSITORY')
    if run_id and repo_name:
        # Use a direct link to the run instead of trying to link to the specific job
        # This is more reliable as the job ID format in URLs is numeric and not available directly
        run_log_url = f"https://github.com/{repo_name}/actions/runs/{run_id}"
        summary_body += f"- [View workflow run log]({run_log_url})\n"

    try:
        pr.create_issue_comment(summary_body)
        print("Successfully created summary comment on PR.")
    except GithubException as e:
        print(f"Error creating summary PR comment: {e}")
    except Exception as e:
        print(f"Unexpected error creating summary PR comment: {e}")
        traceback.print_exc()


def parse_diff_to_patchset(diff_text: str) -> Optional[PatchSet]:
    if not diff_text:
        print("No diff text to parse.")
        return None
    try:
        patch_set = PatchSet(diff_text)
        print(f"Diff parsed into PatchSet with {len(list(patch_set))} patched files.")
        return patch_set
    except Exception as e:
        print(f"Error parsing diff string with unidiff: {type(e).__name__} - {e}")
        print(f"Diff text that failed (first 1000 chars): {diff_text[:1000]}")
    return None


def main():
    print("Starting AI Code Review Script...")
    if not gh or not gemini_client_module:
        print("Error: GitHub or Gemini client not available. Exiting.")
        sys.exit(1)

    pr_details = get_pr_details()
    print(f"Processing PR #{pr_details.pull_number} in repo {pr_details.get_full_repo_name()} (Event: {pr_details.event_type})")

    last_run_sha_from_env = os.environ.get("LAST_RUN_SHA", "").strip()
    head_sha = pr_details.pr_obj.head.sha
    base_sha = pr_details.pr_obj.base.sha

    comparison_sha_for_diff = None
    if pr_details.event_type in ["opened", "reopened"]:
        comparison_sha_for_diff = base_sha
        print(f"Event type is '{pr_details.event_type}'. Reviewing full PR against base SHA: {comparison_sha_for_diff}")
    elif pr_details.event_type == "synchronize":
        if last_run_sha_from_env and last_run_sha_from_env != head_sha :
            comparison_sha_for_diff = last_run_sha_from_env
            print(f"Event type is 'synchronize'. Reviewing changes since last run SHA: {comparison_sha_for_diff}")
        else:
            comparison_sha_for_diff = base_sha
            if not last_run_sha_from_env:
                 print(f"Event type is 'synchronize', but no last_run_sha found. Reviewing full PR against base SHA: {comparison_sha_for_diff}")
            elif last_run_sha_from_env == head_sha:
                 print(f"Event type is 'synchronize', but last_run_sha ({last_run_sha_from_env}) is same as head_sha. No new commits for incremental review. Defaulting to full review against base SHA: {comparison_sha_for_diff}.")
    else:
        comparison_sha_for_diff = base_sha
        print(f"Event type is '{pr_details.event_type}'. Defaulting to full review against base SHA: {comparison_sha_for_diff}")

    if head_sha == comparison_sha_for_diff:
        print(f"HEAD SHA ({head_sha}) is the same as comparison SHA ({comparison_sha_for_diff}). No new changes to diff.")
        save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
        create_review_and_summary_comment(pr_details, [], "reviews/gemini-pr-review.json")
        print("Exiting as there are no new changes to review based on SHAs.")
        return

    diff_text = get_diff(pr_details, comparison_sha_for_diff)
    if not diff_text:
        print("No diff content retrieved. Exiting review process.")
        save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
        create_review_and_summary_comment(pr_details, [], "reviews/gemini-pr-review.json")
        return

    initial_patch_set = parse_diff_to_patchset(diff_text)
    if not initial_patch_set:
        print("Failed to parse diff into PatchSet. Exiting.")
        save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
        sys.exit(1)

    exclude_patterns_str = os.environ.get("INPUT_EXCLUDE", "")
    exclude_patterns = [p.strip() for p in exclude_patterns_str.split(',') if p.strip()]

    actual_files_to_process: List[PatchedFile] = []
    for patched_file_obj in initial_patch_set:
        normalized_path = patched_file_obj.path.lstrip('./')
        is_excluded = False

        if patched_file_obj.is_removed_file or (patched_file_obj.is_added_file and patched_file_obj.target_file == '/dev/null'):
            print(f"Skipping removed file (or added as /dev/null): {patched_file_obj.path}")
            is_excluded = True
        elif patched_file_obj.is_binary_file:
            print(f"Excluding binary file: {patched_file_obj.path}")
            is_excluded = True
        else:
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(normalized_path, pattern) or fnmatch.fnmatch(patched_file_obj.path, pattern):
                    print(f"Excluding file '{patched_file_obj.path}' due to pattern '{pattern}'.")
                    is_excluded = True
                    break
        if not is_excluded:
            actual_files_to_process.append(patched_file_obj)

    num_files_to_analyze = len(actual_files_to_process)
    print(f"Number of files to analyze after exclusions: {num_files_to_analyze}")

    if num_files_to_analyze == 0:
        print("No files to analyze after applying exclusion patterns.")
        save_review_results_to_json(pr_details, [], "reviews/gemini-pr-review.json")
        create_review_and_summary_comment(pr_details, [], "reviews/gemini-pr-review.json")
        return

    comments_for_gh_review_api = analyze_code(actual_files_to_process, pr_details)

    review_json_filepath = "reviews/gemini-pr-review.json"
    save_review_results_to_json(pr_details, comments_for_gh_review_api, review_json_filepath)
    create_review_and_summary_comment(pr_details, comments_for_gh_review_api, review_json_filepath)

    print("AI Code Review Script finished.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Unhandled exception in __main__: {type(e).__name__} - {e}")
        traceback.print_exc()
        sys.exit(1)