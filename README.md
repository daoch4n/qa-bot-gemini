Dont want to pay Copilot to review your code? No problem! Because you've got:
# Gemini PR Reviewer 🧠
## Reviews your Pull Requests for 🆓 using latest Gemini Flash model with previous feedback reevaluation on update ✨

### Usage:
- Put it in `.github/workflows/` folder of your repo
- Go to https://aistudio.google.com/ and obtain Gemini API key there
- From repository page on Github go to `Settings` -> `Secrets and variables` -> `Actions`
- Click `New repository secret`
  - Name: `GEMINI_API_KEY`
  - Secret: the API key you just got from https://aistudio.google.com/
- It will automatically run on every pull request creation , update and reopen

### See it in action 🪄 : [https://github.com/dtub/DaokoTube](https://github.com/dtub/DaokoTube/pulls?q=is%3Apr+is%3Aclosed)
