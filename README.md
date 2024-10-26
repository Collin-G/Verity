# Verity


### **Important Notes**

I did all of my developmment on linux. Although deployment should work fine on windows, please consider WSL if things do not work out. Also, this repo contains the flask API used for processing data and feeding it into neural networks. Our web app was created on repl.it (https://replit.com/@NainAbdi/AirBNBVerifier?v=1)

## **Deployment Instructions**

### **Install Prerequisites**

Install ngrok: https://ngrok.com/download

Install ffmpeg: https://www.ffmpeg.org/download.html

Install python prerequisites: Open a terminal in this repo and type "pip install -r requirements.txt"


### **Setup ngrok to run flask api**

Refer to: https://ngrok.com/docs/getting-started/?os=linux

Complete up to step 2 in the link and then in terminal run: "ngrok http 5000"

Then open a terminal in this repo and type: "gunicorn -w 2 -b 0.0.0.0:5000 --timeout 1000 main:app"

### **Loading web app on repl.it**

Make a repl.it account, and fork the project at this link: "https://replit.com/@NainAbdi/AirBNBVerifier?v=1"

Notice how if you go back to the terminal where you typed "ngrok http 5000", you will see a link that looks simillar to https://e8e0-135-0-212-218.ngrok-free.app.

In the forked repl.it, navigate to /src/app/page.tsx LINE 78 and replace the "https://e8e0-135-0-212-218.ngrok-free.app/predict" on LINE 78 with your link/predict

It will look something like: "https://(your stuff).ngrok-free.app/predict"


Finally, run the app with the button at the top center of the repl.it page!
