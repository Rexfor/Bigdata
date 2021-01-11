# Installing the java JDK.

![1](https://user-images.githubusercontent.com/60914099/104150360-8c587180-538e-11eb-9823-853f118ac155.PNG)

To begin with the installation process we must install the latest version of the Java JDK (Java development kit), its installation is necessary because it is required for the use of spark, this is necessary because it provides us with tools to install and work with various languages different programming and framework.
To find the jdk we enter the following link:
https://www.oracle.com/mx/java/technologies/javase/javase-jdk8-downloads.html

![2](https://user-images.githubusercontent.com/60914099/104150366-8f536200-538e-11eb-8c19-121138321394.png)

On the page we must scroll down where we will have to select the option according to our operating system, on the page you can find installation options for Linux, mac os, solaris and Windows; in this case my installation will proceed with the option for Windows in 64 bits.

![3](https://user-images.githubusercontent.com/60914099/104150375-97130680-538e-11eb-8e22-02e61a5a00d9.png)

When selecting the option to download, the system will show us a window about the Oracle license terms, they must be accepted to start the download.

![4](https://user-images.githubusercontent.com/60914099/104150377-97ab9d00-538e-11eb-9e46-4c70e3a2a414.png)

Once the executable is downloaded, we will open it; After this, a window with the installer will be displayed where we will begin the process for its installation.

![5](https://user-images.githubusercontent.com/60914099/104150380-97ab9d00-538e-11eb-95be-0ca6295fa507.png)

We will proceed to select the “Next” button, a new window will be displayed where we can select the features to install, in this case we will leave the default options by pressing “Next”.

![6](https://user-images.githubusercontent.com/60914099/104150381-98443380-538e-11eb-8487-21f2205dc47f.png)

The necessary files will begin to install.

![7](https://user-images.githubusercontent.com/60914099/104150382-98443380-538e-11eb-9af7-b3167e4ffc73.png)

Once the installer is loaded we will begin to configure java, in this section we will select the destination folder, in this case we will leave the folder created by default and select "Next"

![8](https://user-images.githubusercontent.com/60914099/104150385-98443380-538e-11eb-94a5-9db8352dfa29.png)

Once this is done, Java will begin to install.

![9](https://user-images.githubusercontent.com/60914099/104150386-98443380-538e-11eb-96cb-acb9be2c9874.png)

When we finish loading, we will have installed Java on our computer and we will select "Close" to finish the JDK installation process.

![10](https://user-images.githubusercontent.com/60914099/104150387-98dcca00-538e-11eb-835e-bc786d4a3ef2.png)

Now we must insert Java in the environment variables for the process, to do this, we must do the following.

![11](https://user-images.githubusercontent.com/60914099/104150388-98dcca00-538e-11eb-8421-dd8e4d5e223d.png)

In the search bar, we write "Environment variables". And we select the option that says "Edit the system environment variables".

![12](https://user-images.githubusercontent.com/60914099/104150389-98dcca00-538e-11eb-9933-e469e054f34d.png)

A window like the following one will appear, in this we must select the option of "environment variables"

![13](https://user-images.githubusercontent.com/60914099/104150390-9aa68d80-538e-11eb-8723-be41b9454654.png)

The following window will open where we must select the option that says "NEW", after this the following window will open.

![14](https://user-images.githubusercontent.com/60914099/104150391-99756080-538e-11eb-8528-46870e9c5274.png)

We are going to add the new environment variable to be able to use it.
In the variable name section, we write the following:
JAVA_HOME
In variable value we must select the path where the java file was installed.
And we click on "Accept"

![15](https://user-images.githubusercontent.com/60914099/104150392-99756080-538e-11eb-9ba3-2ab8ce925801.png)

After this we must add the variable in the option "Path" for which we will select it and click on edit.

The following window will open and we must click on "New" and with we will write the following to add the variable to the path
% JAVA_HOME% \ bin
And we will click on accept.

![16](https://user-images.githubusercontent.com/60914099/104150393-99756080-538e-11eb-8a2d-9ef6b10a57bb.png)

We will click on accept to finish the environment variables process.

# Spark installation

![18](https://user-images.githubusercontent.com/60914099/104150396-9a0df700-538e-11eb-86dd-b311ab290b0f.png)

To install spark we must enter the following link:
https://spark.apache.org/downloads.html
In this section we must choose the following values and click on option 3 that says "download spark"

![19](https://user-images.githubusercontent.com/60914099/104150397-9a0df700-538e-11eb-8057-a487699fcd0d.png)

The following window will open in which we select the first option to download the file.

![20](https://user-images.githubusercontent.com/60914099/104150398-9a0df700-538e-11eb-9e47-99089f96bf3d.png)

Once the file is downloaded, we must unzip the file.

![21](https://user-images.githubusercontent.com/60914099/104150399-9aa68d80-538e-11eb-8b20-6184931613a5.png)

Once unzipped, we create a folder where we will save all the files in order to have a path for the environment variable.
Now we must insert spark in the environment variables for the process, to do this, we must do the following.

![22](https://user-images.githubusercontent.com/60914099/104150400-9aa68d80-538e-11eb-8d16-db6640b11837.png)

In the search bar, we write "Environment variables". And we select the option that says "Edit the system environment variables".

![23](https://user-images.githubusercontent.com/60914099/104150402-9aa68d80-538e-11eb-97a9-e0cd00a36570.png)

A window like the following one will appear, in this we must select the option of "environment variables"

![24](https://user-images.githubusercontent.com/60914099/104150403-9aa68d80-538e-11eb-8cdf-8b159ea0f4ef.png)

The following window will open where we must select the option that says "NEW", after this the following window will open.

![25](https://user-images.githubusercontent.com/60914099/104150405-9b3f2400-538e-11eb-8907-a6104473c2a4.png)

We are going to add the new environment variable to be able to use spark in any direction.
In the variable name section, we write the following:
SPARK_HOME
In variable value we must select the path where the spark file was installed.
And we click on "Accept"

![26](https://user-images.githubusercontent.com/60914099/104150406-9b3f2400-538e-11eb-93b3-6a4a07a1108e.png)

After this we must add the variable in the option "Path" for which we will select it and click on edit.

![27](https://user-images.githubusercontent.com/60914099/104150407-9b3f2400-538e-11eb-9d43-1181e282a1ee.png)

The following window will open and we must click on "New" and with we will write the following to add the variable to the path
% SPARK_HOME% \ bin
And we will click on accept.

![28](https://user-images.githubusercontent.com/60914099/104150409-9b3f2400-538e-11eb-9e63-38d176383fe4.png)

We will click on accept to finish the environment variables process.











# Install winutils.exe

![29](https://user-images.githubusercontent.com/60914099/104150410-9bd7ba80-538e-11eb-9759-9918bfb95961.png)

To install winutil.exe we must enter the following github link
https://github.com/cdarlint/winutils
In this github we must look for the file according to the version of spark that we installed.

![30](https://user-images.githubusercontent.com/60914099/104150411-9bd7ba80-538e-11eb-862b-3cf85904e12d.png)

We will enter and look for the winutil.exe file and click, after this another tab will open, in which the "download" button will appear to download the file.

![31](https://user-images.githubusercontent.com/60914099/104150412-9bd7ba80-538e-11eb-853a-1cc5751e75c1.png)

Once downloaded we must create a folder in the root (c :) with the name of winutils, within this folder we must create another folder called bin and we must leave the downloaded file.
Now we must insert HADOOP in the environment variables for the process, to do this, we must do the following.

![32](https://user-images.githubusercontent.com/60914099/104150414-9bd7ba80-538e-11eb-93ff-a52cf3c2c78a.png)

In the search bar, we write "Environment variables". And we select the option that says "Edit the system environment variables".

![33](https://user-images.githubusercontent.com/60914099/104150415-9c705100-538e-11eb-824a-10004b28cd0b.png)

A window like the following one will appear, in this we must select the option of "environment variables"

![34](https://user-images.githubusercontent.com/60914099/104150417-9c705100-538e-11eb-8ab6-bd4c8eb9b642.png)

The following window will open where we must select the option that says "NEW", after this the following window will open.

![35](https://user-images.githubusercontent.com/60914099/104150418-9c705100-538e-11eb-91fc-d7a52e52dee5.png)

We are going to add the new environment variable to be able to use HADOOP in any direction.
In the variable name section, we write the following:
HADOOP_HOME
In variable value we must select the path where the HADOOP file was installed.
And we click on "Accept"

![36](https://user-images.githubusercontent.com/60914099/104150419-9c705100-538e-11eb-9990-97864c5fd9a0.png)

After this we must add the variable in the option "Path" for which we will select it and click on edit.

![37](https://user-images.githubusercontent.com/60914099/104150420-9d08e780-538e-11eb-9420-2a51545e6f10.png)

The following window will open and we must click on "New" and with we will write the following to add the variable to the path
% HADOOP_HOME% \ bin
And we will click on accept.

![38](https://user-images.githubusercontent.com/60914099/104150421-9d08e780-538e-11eb-998c-ef38fd785f9f.png)

We will click on accept to finish the environment variables process.

# Install scala

![39](https://user-images.githubusercontent.com/60914099/104150423-9d08e780-538e-11eb-98ac-2f18b2c41ba0.png)

To install scala we must go to the following link:
http://scala-ide.org/download/sdk.html
Here we must click on the button "Download IDE"

![40](https://user-images.githubusercontent.com/60914099/104150424-9da17e00-538e-11eb-933e-fed135fbf63c.png)

Once downloaded we must unzip it in a desired path.

![41](https://user-images.githubusercontent.com/60914099/104150425-9da17e00-538e-11eb-96de-254acce6126e.png)

Once the above is ready, we open a terminal, typing cmd, we proceed to change directory to enter the folder where we install spark, in my case it is in c: \ spark, once this is done we are going to write the spark-shell command to enter the spark environment.

![42](https://user-images.githubusercontent.com/60914099/104150426-9da17e00-538e-11eb-9e84-b8c643816904.png)

If everything was done correctly, we will have scala and spark installed on Windows.

# Author
* **Rexfor** - [github] (https://github.com/Rexfor)
