Don't want to pay Copilot to review your code online? No problem! Because you've got:
# 🪭 Zen AI QA ✨
## Reviews your Pull Requests for 🆓 using latest Gemini Flash model with previous feedback reevaluation on update ✨

### Usage:
- Put both `.py` and `.yml` files in `.github/workflows/` folder of your repo
- Go to https://aistudio.google.com/ and obtain Gemini API key there
- From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
  - Click `New repository secret`
    - Name: `GEMINI_API_KEY`
    - Secret: the API key you just got from https://aistudio.google.com/
- It will automatically run on every pull request creation , update and reopen
- Detailed review feedback will be generated on completion with `Resolve conversation` button <br> along with AI-actionable JSON report (in two formats) <br> auto-commited by cute white robot to your repo `/reviews/` folder
### (Optional) If you also want it to comment as cute white robot:
- Install https://github.com/apps/zen-ai-qa on your acc or repo
- From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
  - Click `New repository secret`
    - Set `ZEN_DEVOPS_APP_INSTALLATION_ID` with the installation ID <br> (you can find installation ID in url of app settings page after you install app)
    - Set `ZEN_DEVOPS_APP_PRIVATE_KEY` (mailto:daoch4n@xn--vck1b.shop)
- Or make you own app in Developer settings and use its installation ID and key! 🗝️

### See it in action 🪄 : [dtub/DaokoTube](https://github.com/dtub/DaokoTube/pulls?q=is%3Apr+is%3Aclosed)

Inspired by [truongnh1992/gemini-ai-code-reviewer](https://github.com/truongnh1992/gemini-ai-code-reviewer)
<br><br>
Differences:
- Batches hunks related to same files to optimize rate limiting
- Makes use of Gemini 1 million tokens context window <br> by attaching whole file together with changes for better context
- Optimized diff parsing algoritm that work better with GitHub (maybe)
- Uses Structured Output mode of Gemini API for better parsing of AI output
- Rotates API keys through `GEMINI_ALT_1` >> `GEMINI_ALT_4` (dont forget to set them in repos secrets) <br> in case your PRs are really big and active
- Auto-commits AI-actionable JSON report to your repo /reviews/ folder for futher agentic processing
- Uses JSON file in question during PR update run to load previous comments from last run <br> for better context during subsequent runs
