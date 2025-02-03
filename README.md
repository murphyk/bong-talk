# BONG: Bayesian Online Natural Gradient

## Run locally
To start the slide show:

- clone this repo
- `npm install`
- `npm run dev` or `npx slidev`
- visit <http://localhost:3030>

## To build a new version

Settings>Pages>Build and deployment
  Change source to github actions

Go to .github/workflows/deploy.yml
Change line 28 to '--base /reponame/'

For testing:
'git commit -m "[skip actions] foo foo"`