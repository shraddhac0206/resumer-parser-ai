# Step-by-Step: Push to GitHub

## Step 1: Create Repository on GitHub

1. Go to **https://github.com/new**
2. Sign in to your GitHub account
3. Fill in:
   - **Repository name**: `resume-parser-ai` (or any name you like)
   - **Description**: "AI-powered resume parser with web interface"
   - **Visibility**: Choose Public or Private
   - **IMPORTANT**: Leave all checkboxes UNCHECKED (don't add README, .gitignore, or license)
4. Click **"Create repository"**

## Step 2: Copy Your Repository URL

After creating, GitHub will show you a page with setup instructions. You'll see a URL like:
- `https://github.com/YOUR_USERNAME/resume-parser-ai.git`

**Copy this URL** - you'll need it in the next step.

## Step 3: Connect and Push (Run These Commands)

Open PowerShell or Command Prompt in your project folder, then run:

```powershell
# Make sure you're in the right directory
cd "C:\Users\shrad\Desktop\Resume Parser"

# Add the remote repository (replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/resume-parser-ai.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Authentication

When you run `git push`, you'll be prompted for credentials:

### Option A: Personal Access Token (Recommended)

1. If prompted for username: Enter your GitHub username
2. If prompted for password: **Don't use your GitHub password!**
   - Instead, use a **Personal Access Token**
   
3. **Create a Personal Access Token:**
   - Go to: https://github.com/settings/tokens
   - Click **"Generate new token"** → **"Generate new token (classic)"**
   - Give it a name: "Resume Parser Project"
   - Select expiration (30 days, 90 days, or no expiration)
   - Check the **"repo"** scope (this gives access to repositories)
   - Click **"Generate token"**
   - **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
   - Paste this token as your password when prompted

### Option B: GitHub CLI (Easier Alternative)

If you have GitHub CLI installed:
```powershell
gh auth login
git push -u origin main
```

## Complete Example

Here's what the full sequence looks like:

```powershell
# Navigate to project
cd "C:\Users\shrad\Desktop\Resume Parser"

# Add remote (replace 'yourusername' with your actual GitHub username)
git remote add origin https://github.com/yourusername/resume-parser-ai.git

# Set main branch
git branch -M main

# Push (you'll be prompted for credentials)
git push -u origin main
```

## Troubleshooting

### "remote origin already exists"
If you get this error, remove the old remote first:
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/resume-parser-ai.git
```

### "Authentication failed"
- Make sure you're using a Personal Access Token, not your password
- Check that the token has "repo" scope enabled
- Try creating a new token if the old one expired

### "Repository not found"
- Double-check the repository name and your username
- Make sure the repository exists on GitHub
- Verify you have access to the repository

### "Permission denied"
- Make sure you're signed in to the correct GitHub account
- Check that the repository name matches exactly

## Verify It Worked

1. Go to your GitHub repository page: `https://github.com/YOUR_USERNAME/resume-parser-ai`
2. You should see all your files there
3. The README.md will be displayed on the homepage

## Future Updates

To push future changes:
```powershell
git add .
git commit -m "Description of changes"
git push
```

---

**Need help?** Check the error message and try the troubleshooting steps above!
