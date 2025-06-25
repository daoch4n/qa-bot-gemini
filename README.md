# 👁️‍🗨️ zen-ai-qa[bot] 👀

### Run AI Code Review on pushes to main branch and Pull Requests using Gemini Flash 2.5 model with previous feedback reevaluation ✨

### [⬇️ Skip to Readme](#readme)

![image](https://github.com/user-attachments/assets/a3c2b16f-7339-4b25-bb14-ccb291b2fcc0)

## Features
- Automatically runs <br> on pushes to main branch and Pull Requests (creation, update , reopen)
- Talks to you as your custom [bot] <br> or as github-actions[bot] if you didn't set it up
- Auto-commits AI-actionable JSON report as your custom [bot] <br> to your repo `/reviews/` folder for futher agentic processing <br> or as zen-ai-dev[bot] if you didn't update the hardcoded git settings in `.yml` files
- Uses JSON file in question during next run <br> for better context and logical consistency
- Batches hunks related to same files <br> to optimize rate limiting
- Makes use of Gemini 1 million tokens context window <br> by attaching whole file together with changes for better context
- Optimized diff parsing algoritm <br> that works better with GitHub (maybe?)
- Uses Structured Output mode of Gemini API <br> for better parsing of AI output
- Compares actual changes to commit titles <br> on pushes to main branch only

![image](https://github.com/user-attachments/assets/64436b3d-4166-4ae5-be0e-6320088981c4)

## Readme
- Put `.py` and `.yml` files in `.github/workflows/` folder of your repo
- Go to [AI Studio](https://aistudio.google.com/apikey) and obtain Gemini API key there
- From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
  - Click `New repository secret`
    - Name: `GEMINI_API_KEY`
    - Secret: the API key you just got from [AI Studio](https://aistudio.google.com/apikey) in JSON format: [ key ]
- It will automatically run on pushes to main branch and pull request creation , update and reopen
- For pushes to main branch, detailed review feedback since last push will be auto-commited to your repo `/reviews/` folder (separate from PR feedback)
- For Pull Requests, detailed review feedback will be commented by your custom [bot] on completion <br> auto-commited to same folder (separate from main branch commits feedback)
### (Optional) <br> If you also want it to comment as your custom bot:
- Make you own app in Developer settings and use its installation ID and key! 🗝️
- From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
  - Click `New repository secret`
    - Set `ZEN_APP_INSTALLATION_ID` with the installation ID <br> you can find installation ID in url of app settings page (the one that displays after you install app on your account or org, not the one where you generate private key)
    - Set `ZEN_APP_PRIVATE_KEY` with your app private key generated in [app settings](https://github.com/settings/apps/)

Inspired by [truongnh1992/gemini-ai-code-reviewer](https://github.com/truongnh1992/gemini-ai-code-reviewer)

