# ML_PROJECT_MSC_AI
Semester Project for the Machine Learning course of MSc AI

## SETUP

## Flask Model Service Deployment Instructions

### Step 1: Install Docker

If you haven't already, install Docker on your computer by following the instructions on the official Docker website: [Get Docker](https://docs.docker.com/get-docker/)

### Step 2: Build the container

docker build -t <image_name> .

### Step 3: Run the Docker Container

docker run -p 8000:8000 <image_name>

### Step 4: Access the Flask App

http://localhost:8000


### Step 5: input test csv file
