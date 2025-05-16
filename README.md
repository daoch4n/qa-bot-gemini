Don't want to pay Copilot to review your code online? No problem! Because you've got:
# Gemini PR Reviewer 🧠
## Reviews your Pull Requests for 🆓 using latest Gemini Flash model with previous feedback reevaluation on update ✨

### Usage:
- Put both `.py` and `.yml` files in `.github/workflows/` folder of your repo
- Go to https://aistudio.google.com/ and obtain Gemini API key there
- From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
- Click `New repository secret`
  - Name: `GEMINI_API_KEY`
  - Secret: the API key you just got from https://aistudio.google.com/
- It will automatically run on every pull request creation , update and reopen
- Detailed review feedback will be generated on completion with `Resolve conversation` button along with AI-actionable JSON report auto-commited to your repo `/reviews/` folder in case you'd like to delegate bugfixing to another agent
- Also uses JSON file in question to load previous comments from last run for better context
### (Optional) If you want it to comment as cute white robot:
- Install https://github.com/apps/zen-ai-qa on your acc or repo
  - From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
    - Click `New repository secret`
    - Set `ZEN_DEVOPS_APP_INSTALLATION_ID` with the installation ID (you can find installation ID in url of app settings page after you install app)
    - Set `ZEN_DEVOPS_APP_PRIVATE_KEY` 🚧

### See it in action 🪄 : [https://github.com/dtub/DaokoTube](https://github.com/dtub/DaokoTube/pulls?q=is%3Apr+is%3Aclosed)
### Inspired by https://github.com/truongnh1992/gemini-ai-code-reviewer
Differences:
- Batches hunks related to same files to optimize rate limiting
- Makes use of Gemini 1 million tokens context window by attaching whole file together with changes for better context
- Loads previous comments on PR update for better context
- Optimized diff parsing algoritm that work better with GitHub (maybe)
- Uses Structured Output mode of Gemini API for better parsing of AI output
