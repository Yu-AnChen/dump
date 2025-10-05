# Palomero Web Environment Setup Guide for Windows

This guide provides step-by-step instructions for setting up and running the
Palomero web application on a Windows machine using the provided Pixi
environment.

## Prerequisites

* A Windows 10 or Windows 11 computer.
* Your HMS eCommons credentials.
* The palomero-env.zip file.

**Important Note:** These instructions are **for Windows machines only**. If you
are using macOS or Linux, please reach out to **Yu-An** or refer to the
[palomero GitHub repository's README for
guidance](https://github.com/yu-anchen/palomero).

## Step 1: Unzip the Environment

First, place the environment files in a memorable location on your computer.
Your user home directory is a great choice.

1. Open **File Explorer** and navigate to your user folder (e.g.,
   `C:\Users\<your-username>`).
2. Copy the **palomero-env.zip** file into this directory.
3. Right-click on **palomero-env.zip** and select **"Extract All..."**.
4. Ensure the final folder path is `...\palomero-env`, not
   `...\palomero-env\palomero-env`. Some unzip tools create a nested folder; if
   this happens, move the contents up one level.

Inside this directory, you will find the Pixi executable (**pixi.exe**) and the
project's lock file (**pixi.lock**), which precisely defines the software
environment.

## Step 2: Open a Command Prompt

All commands must be run from within the palomero-env directory. The easiest way
to do this is:

1. In File Explorer, navigate into the `palomero-env` folder.
2. Click in the address bar at the top of the window.
3. Type **cmd** and press **Enter**.

This will open a new Command Prompt window with the correct path already set.

## Step 3: Install Dependencies

Now, you will install the environment. This command reads the **pixi.lock** file
and downloads all the necessary packages.

In the Command Prompt window, run the following command:

```cmd
pixi.exe run palomero-web --help
```

**Note:** This installation step can take several minutes, especially on the
first run. Once it finishes, it will display a help message for the palomero-web
command, confirming that the environment is set up correctly.

## Step 4: Log into OMERO

Next, authenticate your session with the OMERO server. Run the command below,
replacing `<your-ecommons>` with your actual eCommons ID.

```cmd
pixi.exe run omero login -s omero-app.hms.harvard.edu -u <your-ecommons> -p 4064 -t 999999
```

You will be prompted to enter your password. Type it and press Enter. (Your
password will not be visible as you type).

## Step 5: Launch the Web Server

With setup and login complete, start the local web server.

```cmd
pixi.exe run palomero-web
```

The server will start, and you will see output in your terminal indicating that
it is running on **<http://localhost:5001>**.

**Important:** Do not close this Command Prompt window. Closing it will shut
down the server.

## Step 6: Access Palomero in Your Browser

1. Open your preferred web browser (Chrome, Firefox, or Edge).
2. Navigate to this address: **<http://localhost:5001>**

The Palomero web interface should load and be ready for use.

## How to Shut Down the Server

To stop the server, return to the open Command Prompt window and press **Ctrl +
C** on your keyboard.

## How to Uninstall Palomero Web Environment

Uninstalling Palomero is simple and does not require any special tools.

1. **Shut Down the Server**  
   If the Palomero web server is running, return to the Command Prompt window
   and press **Ctrl + C** to stop it.

2. **Close Windows Accessing the Folder**  
   Make sure to close any Command Prompt or File Explorer windows that are
   currently open in the `palomero-env` folder.  
   This ensures the folder is not locked and can be deleted without errors.

3. **Delete the Environment Folder**  
   In File Explorer, navigate to the location where you extracted
   `palomero-env`.  
   Right-click the `palomero-env` folder and select **Delete**.

## Troubleshooting

* **'pixi.exe' is not recognized... error:** This error means your Command
  Prompt is not in the correct directory. Close the Command Prompt and repeat
  **Step 2** carefully.
* **Login Failed:** Double-check that you replaced `<your-ecommons>` with your
  correct ID and that you typed your password correctly.
* **"Address already in use" error:** This means another application is using
  port 5001. You can choose a different port, for example:

  ```cmd
  pixi.exe run palomero-web --port 9001
  ```
