# 👁️‍🗨️ zen-ai-qa[bot]

### Run Gemini AI Code Review on pushes to main branch and pull requests with previous feedback reevaluation

### [⬇️ Skip to Readme](#readme)

![image](https://github.com/user-attachments/assets/a3c2b16f-7339-4b25-bb14-ccb291b2fcc0)

## Features
- Automatically runs <br> on pushes to main branch and pull requests
- Posts review comments as custom [bot]
- Auto-commits AI-actionable JSON report <br> to your repo `/reviews/` folder for futher agentic processing
- Reloads JSON report in question during next run
- Batches hunks related to same files
- Optimized diff parsing algoritm

![image](https://github.com/user-attachments/assets/64436b3d-4166-4ae5-be0e-6320088981c4)

## Readme
- Put `.py` and `.yml` files in `.github/workflows/` folder of your repo
- Go to [AI Studio](https://aistudio.google.com/apikey) and obtain Gemini API key there
- On the repo page where you want to run this bot <br> Go to `Settings` -> `Secrets and variables` -> `Actions`
  - Click `New repository secret`
    - Name: `GEMINI_API_KEY`
    - Secret: the API key you just got from [AI Studio](https://aistudio.google.com/apikey) in JSON format: [ key ]
- It will automatically run on pushes to main branch and pull request creation , update and reopen
- For pushes to main branch, detailed review feedback since last push will be auto-commited to your repo `/reviews/` folder (separate from PR feedback)
- For Pull Requests, detailed review feedback will be commented by your custom [bot] on completion <br> and auto-commited to same folder (separate from main branch commits feedback)
### (Optional) <br> If you also want it to comment as your custom bot:
- Make your own app in Developer settings and use its installation ID and key! 🗝️
- From repo page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
  - Click `New repository secret`
    - Set `ZEN_APP_INSTALLATION_ID` with the installation ID <br> you can find installation ID in url of app settings page (the one that displays after you install app on your account or org, not the one where you generate private key)
    - Set `ZEN_APP_PRIVATE_KEY` with your app private key generated in [app settings](https://github.com/settings/apps/)
    - Replace hardcoded App ID in `.py` file with your own app ID <!-- TODO: Refactor script to load App ID from Actions Secrets  -->
<br><br>
Inspired by [truongnh1992/gemini-ai-code-reviewer](https://github.com/truongnh1992/gemini-ai-code-reviewer)

