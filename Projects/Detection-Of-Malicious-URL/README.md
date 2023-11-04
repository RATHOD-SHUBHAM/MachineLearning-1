# URLcheck

Learning based Malicious Web Sites Detection

**ABSTRACT**

Malicious Web sites largely promote the growth of Internet criminal activities and constrain the development of Web services. As a result, there has been strong motivation to develop systemic solution to stopping the user from visiting such Web sites. 

We propose a learning based approach to classifying Web sites into 3 classes: Benign, Spam and Malicious.

Our mechanism only analyzes the Uniform Resource Locator (URL) itself without accessing the content of Web sites. 
Thus, it eliminates the run-time latency and the possibility of exposing users to the browser based vulnerabilities.
By employing learning algorithms, our scheme achieves better performance on generality and coverage compared with blacklisting service. 


## PROJECT APPROACH

URLs of the websites are separated into 3 classes:

* Benign: Safe websites with normal services
* Spam: Website performs the act of attempting to flood the user with advertising or sites such as fake surveys and online dating etc.
* Malware: Website created by attackers to disrupt computer operation, gather sensitive information, or gain access to private computer systems.



### Feature Extraction
Given single URL, we extract its features and  categorize them into 3 classes:



**1. Lexical Features**

**2. Site popularity Features**
 
**3. Host-based Features**


### Training

We used two supervised learning algorithms **random forest** and **support vector machine** 
