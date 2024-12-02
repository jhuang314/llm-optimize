# llm-optimize

Deploy your own Large Language Models compatible with OpenAI API onto the Akash Network!

0. [Quickstart Docker Build](#quickstart-docker-build)
1. [Quickstart Deployment to Akash Network](#quickstart-deployment-to-akash-network)
2. [Running locally without docker](#running-locally-without-docker)
3. [Running locally using docker compose](#running-locally-using-docker-compose)
4. [Important environment variables](#important-env-vars)
5. [Technical Details](#technical-details)

## Quickstart Docker Build

Here are the instructions for building your own Docker image from this repository and pushing it to Docker Hub.

**Step 1 (Install dependencies)**

Install the following programs if necessary:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- git

Clone this repository:

```bash
git clone https://github.com/jhuang314/llm-optimize
cd llm-optimize
```

Build the docker image:

```bash
docker build -t my-username/my-image:my-tag .

# An explicit example:
docker build -t jh3141/llm-optimize:0.0.2 .
```

Push the docker image to docker hub

```bash
docker push my-username/my-image:my-tag

# An explicit example:
docker push jh3141/llm-optimize:0.0.2
```

## Quickstart Deployment to Akash Network

Here are the instructions for taking a Docker image and deploying it to the Akash Network!

### Step 1 (Copy the YAML SDL config)

Here are 2 sample YAML SDL's, depending on if you want to use CPU's vs GPU's. Obviously, the GPUs are much faster.

The actual models can be modified in [`main.py`](https://github.com/jhuang314/llm-optimize/blob/main/main.py#L26-L31).

For either SDL, replace `<YOUR_HF_TOKEN>` with your actual [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens).

If you built and pushed your own Docker image, feel free to replace `jh3141/llm-optimize:0.0.2` with your own.

**CPU version**
```yaml
---
version: "2.0"
services:
  web:
    image: jh3141/llm-optimize:0.0.2
    expose:
      - port: 8000
        as: 80
        to:
          - global: true
    env:
      - HF_TOKEN=<YOUR_HF_TOKEN>
profiles:
  compute:
    web:
      resources:
        cpu:
          units: 16
        memory:
          size: 32Gi
        storage:
          - size: 30Gi
  placement:
    akash:
      pricing:
        web:
          denom: uakt
          amount: 10000
deployment:
  web:
    akash:
      profile: web
      count: 1
```


**GPU version**
```yaml
---
version: "2.0"
services:
  web:
    image: jh3141/llm-optimize:0.0.2
    expose:
      - port: 5000
        as: 80
        to:
          - global: true
    env:
      - HF_TOKEN=<YOUR_HF_TOKEN>
profiles:
  compute:
    web:
      resources:
        cpu:
          units: 16
        memory:
          size: 32Gi
        storage:
          - size: 30Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
  placement:
    akash:
      pricing:
        web:
          denom: uakt
          amount: 10000
deployment:
  web:
    akash:
      profile: web
      count: 1
```


### Step 2 (Deploy to Akash console)

1. Go to the Akash console: https://console.akash.network/, and click on Deploy. Feel free to activate the $10 trial to get some funds.
1. Click on Deploy
1. Choose "Run Custom Container"
1. Switch from "Builder" tab to "YAML" tab
1. Paste the whole YAML SDL file from above. (Be sure to add your HF_TOKEN)
1. Click "Create Deployment ->", and Confirm
1. Pick a provider, and "Accept Bid ->"
1. Wait a bit
1. On the "Leases" tab, open the URI(s) link
1. Enjoy using Akash Image Generator, powered by Akash Network!


## Running Locally (without Docker)


### Step 1 (clone this repo if you haven't already):

```bash
git clone https://github.com/jhuang314/llm-optimize
cd llm-optimize
```

### Step 2 (update .env):

Create a `.env` file within the base directory of this repo:

```bash
touch .env
echo 'HF_TOKEN=<YOUR_HF_TOKEN>' >> .env
```

### Step 3 (install dependencies)

The following commands assumes you are in the base directory of this repo.


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4 (run the api):

The following commands assumes you are in the base directory of this repo.



**Starting the Flask server**

If you don't already have virtual environment setup in step 3, do that step.

Otherwise, activate the environment if you haven't already:

```bash
cd backend
source .venv/bin/activate
```

```bash
gunicorn -w 1 -b 0.0.0.0 main:app
```




## Important env vars

You can obtain your own [`HF_TOKEN` here](https://huggingface.co/docs/hub/en/security-tokens)

```
HF_TOKEN=<hugging face token used to download transformer models>
```


## Technical Details

### Flask API

The `main.py` provies a subset of OpenAI compatible API for LLMs. In a nutshell:

 - POST /v1/completions provides the completion for a single prompt
 - POST /v1/chat/completions provides the response for a given dialog
