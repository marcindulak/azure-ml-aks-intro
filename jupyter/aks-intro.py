#!/usr/bin/env python
# coding: utf-8

# # Introduction to Azure Machine Learning model deployment on Azure Kubernetes Service

# The notebook starts with a short introduction to several Kubernetes concepts: pods, deployments and services.
# 
# Then a machine learning model in created locally in the notebook, based on the exercise from https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks/student-admissions
# 
# The model is then deployed using Azure ML (Machine Learning) to an AKS (Azure Kubernetes Service) cluster created in advance.
# 
# The notebook ends with an exploration of the resources created on the Kubernetes cluster in relation to the prediction endpoint.

# # Warning! One should avoid using Jupyter notebooks to train and deploy ML models. Use notebooks for exploration or teaching purposes!

# # 0. Introduction to Kubernetes

# https://github.com/kubernetes/kubernetes is an open-source container orchestration platform. It helps to configure, start and stop containers. Kubernetes is a production-grade system that offers high-availability, autoscaling, self-healing of the workloads and more. The main focus of Kubernetes are web services served over HTTP and other web protocols, but Kubernetes is flexible enough to be used for hosting and serving data science pipelines, for example using https://github.com/kubeflow/kubeflow, Azure Machine Learning, AWS Sagemaker, and others.
# 
# A container is a name for isolating operating system processes. This means that a process which is containerized has access to only selected and limited resources or fractions of those, provided by the operating system, like CPU, GPU, storage, network. Kubernetes allows one to combine more than one container into a unit called "pod". Pod is a collection of containers that share resources.
# 
# A group of functionally identical pods is controlled by so called "deployment". Deployment can be used to adjust the number of pod replicas to the system load, automatically restart failed pods, and more.
# 
# There several ways of making an application running inside of a pod in a Kubernetes cluster accessible from the internet. One of such ways is called "NodePort" service, which exposes a port on every node (server, physical or virtual) that is part of the Kubernetes cluster, and forwards the network traffic on this port to the port running the application in the pod.
# 
# The process of setting up Kubernetes is involved but for experimenting it can be setup on e.g. https://www.docker.com/products/docker-desktop, https://kind.sigs.k8s.io/ or https://microk8s.io/.
# 
# After the Kubernetes cluster is setup the basic way of interacting with it is by using the https://kubernetes.io/docs/tasks/tools/install-kubectl/ command line client. Examples of kubectl usage are provided at https://kubernetes.io/docs/reference/kubectl/cheatsheet/

# ## 0.1 Add kubectl to PATH

# The Kubernetes cluster is assumed to already exist, and kubectl and kubeconfig available somewhere in the path accessible from this Jupyter notebook instance.

# In[1]:


import os
HOME = os.environ['HOME']


# In[2]:


os.environ["PATH"] += os.pathsep + f"{HOME}/work/bin"


# In[3]:


get_ipython().system('kubectl version --client')


# ## 0.2 Set KUBECONFIG environment variable to point to the AKS cluster config

# In[4]:


os.environ["KUBECONFIG"] = f"{HOME}/work/kubeconfig.yaml"


# In[5]:


get_ipython().system('kubectl version')


# ## 0.3 Kubernetes Nodes

# The Kubernetes cluster consists of master components that control the overall functionality of the cluster (so called "control-plane"), and worker nodes, in this case virtual machines, which execute workloads. An example workload could be an ML model prediction HTTP service.
# 
# The image below, taken from https://kubernetes.io/docs/concepts/overview/components/, shows the Kubernetes control-plane components on the left and three Kubernetes nodes (workers) on the right.
# 
# ![Kubernetes Components](https://d33wubrfki0l68.cloudfront.net/7016517375d10c702489167e704dcb99e570df85/7bb53/images/docs/components-of-kubernetes.png)

# In[6]:


get_ipython().system('kubectl get node')


# ## 0.4 Kubernetes Pods

# Several namespaces are created on a Kubernetes cluster by default.

# In[7]:


get_ipython().system('kubectl get namespaces')


# List the currently running pods. There are no pods running in the "default" namespace, but there are some pods running in the "kube-system" namespace which contains for example some Azure specific pods related to networking and the coredns pods which are responsible for DNS resolution in the Kubernetes cluster.

# In[8]:


get_ipython().system('kubectl get pod')


# In[9]:


get_ipython().system('kubectl -n kube-system get pod')


# Let's create an Nginx (a webserver) https://nginx.org/en/ pod in the default namespace.
# 
# Containers, which are the building blocks of Kubernetes pods, come in different formats. The most popular format for containers is https://www.docker.com/. The definition of a container is stored in so called image. Let's use the nginx Docker image hosted on https://hub.docker.com/_/nginx

# In[10]:


get_ipython().system('kubectl run nginx --image=nginx --restart=Never')


# In[11]:


get_ipython().system('kubectl get pod')


# The nginx pod is in initializing "ContainerCreating" state. In order to start it needs to pull Nginx Docker image from Docker Hub.

# In[12]:


get_ipython().system('kubectl describe pod nginx | grep Normal')


# After some time the nginx pod should be in Running state. This behavior of waiting for resources to be ready illustrates the design choice made by Kubernetes. Distributed systems, which communicate over network, need to be resilient to network failures and delays. That's why kubectl accepted the command to run the nginx pod, but the operations needed to bring up the pod were performed in the background, and would be retrying in case of failures.

# In[13]:


get_ipython().system('sleep 20')
get_ipython().system('kubectl get pod nginx')


# Let's enter ("exec") into the pod and list the running processes. Docker images are typically small and don't contain many tools, therefore the `procps` package which provides the `ps` command utility may need to be installed inside of the container, if the utility was missing in the image.
# 
# Notice that the number of processes is smaller than a typical number of the processes on a typical Linux machine. Note also that process ID 1 is actually the Nginx itself.
# 
# The command below is a typical example of interacting with kubectl client: complex queries can be performed on the objects present in Kubernetes like pods using standard Linux tools like subshells. This is one of the strong points of Kubernetes: it provides the basic operational functions without the need of writing custom scripts.

# In[14]:


get_ipython().system("kubectl exec -it $(kubectl get pod -lrun=nginx -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'apt-get update && apt-get install -y procps'")


# In[15]:


get_ipython().system("kubectl exec -it $(kubectl get pod -lrun=nginx -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'ps aux'")


# One can verify that the Nginx is serving the default web page on port 80 inside of the container.

# In[16]:


get_ipython().system("kubectl exec -it $(kubectl get pod -lrun=nginx -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'curl localhost:80'")


# Let's delete the pod. After a couple of seconds the pod will be deleted.

# In[17]:


get_ipython().system('kubectl delete pod nginx')


# In[18]:


get_ipython().system('kubectl get pod')


# # 0.5 Kubernetes Deployments

# Another type of Kubernetes resource is a deployment. Deployment consists of a specified number of pods, and Kubernetes makes sure that the requested number of pods is running.

# Let's create a deployment in the default namespace consisting of a single Nginx pod.

# In[19]:


get_ipython().system('kubectl create deployment nginx --image=nginx')


# In[20]:


get_ipython().system('kubectl get deployment nginx')


# In[21]:


get_ipython().system('sleep 10')
get_ipython().system('kubectl get pod')


# Notice the name of the pod created by the deployment: it's the requested name `nginx` followed by so called replica set identifier, followed by the pod identifier.

# Let's scale up the deployment to contain two pods

# In[22]:


get_ipython().system('kubectl scale --replicas=2 deployment/nginx')


# In[23]:


get_ipython().system('sleep 10')
get_ipython().system('kubectl get pod')


# Let's delete the first pod belonging to the deployment

# In[24]:


get_ipython().system("kubectl delete pod $(kubectl get pod -lapp=nginx -o jsonpath='{.items[0].metadata.name}')")


# In[25]:


get_ipython().system('kubectl get pod')


# After a while the nginx pod belonging to the deployment is started again, as shown by the younger AGE of the first pod. This is expected since the scaled-up deployment requires now two pods to be running.

# In[26]:


get_ipython().system('sleep 10')
get_ipython().system('kubectl get pod')


# ## 0.6 Kubernetes Services

# The Nginx webserver is accessible from within the Kubernetes cluster network range. One would like to make it accessible from the internet, for example for serving ML predictions over HTTP. One of the ways Kubernetes exposes applications running in pods is by using "NodePort" service.
# 
# The name "NodePort" originates from the way this feature functions: a port is opened on all the worker nodes present in the Kubernetes cluster, and the network traffic is forwarded the to the pods.
# 
# By default security settings don't allow the traffic from the internet to directly reach the nodeports on the nodes. A loadbalancer is typically placed in front and passes the traffic to the nodeports instead. The configuration of a loadbalancer is not demonstrated here.

# In[27]:


get_ipython().system('kubectl expose deployment nginx --name=nginx --type=NodePort --port=80 --target-port=80')


# In[28]:


get_ipython().system('kubectl get service nginx')


# The Nginx welcome page served by the pod is available on the randomly assigned nodeport number from the range 30000-32767. This can be verified by starting a short-lived Nginx pod (called "curl" in this case) and using curl. In this case the traffic from the "curl" pod to the nodeport on the node is allowed.

# In[29]:


get_ipython().run_cell_magic('bash', '', 'NODE=$(kubectl get nodes \\\n       -o jsonpath=\'{ $.items[*].status.addresses[?(@.type=="InternalIP")].address}\')\nNODEPORT=$(kubectl get service nginx -o jsonpath="{.spec.ports[0].nodePort}")\n\necho curl -s $NODE:$NODEPORT\nkubectl run -it --rm curl --image=nginx --restart=Never \\\n        -- bash -c "curl -s $NODE:$NODEPORT"')


# Let's delete the service and deployment.

# In[30]:


get_ipython().system('kubectl delete service nginx')


# In[31]:


get_ipython().system('kubectl delete deployment nginx')


# ## This concludes the overview of the Kubernetes concepts needed in order to follow the notebook. There are more concepts, like Kubernetes storage, but they are omitted here.

# # 1. Create the model

# ## Predicting Student Admissions
# 
# In this notebook, student admissions to graduate school at UCLA are predicted based on three pieces of data:
# - GRE Scores (Test) (200-800, 800 is best)
# - GPA Scores (Grades) (0-4, 4 is best)
# - Institution rank (1-4, 1 is best). The original udacity notebook uses the term "Class", but this is not a "Class", but an "Institution" rank, see the section below for details
# 
# The dataset originally came from here: http://www.ats.ucla.edu/, most likely from https://stats.idre.ucla.edu/stata/dae/logistic-regression/

# ## Explanations of the terms used in the dataset
# 
# 
# Graduate Record Examinations (GRE) is a standardized test in the United States and Canada used for admissions to graduate (master's and doctoral) schools. https://en.wikipedia.org/wiki/Graduate_Record_Examinations. The data in this notebook predates the GRE changes in August 2011 when the scoring scale was changed from a from a 200 to 800 scale to a 130 to 170 scale.
# 
# Grade point average (GPA) is calculated by averaging the grade points a student earns in a given period of time. GPA ranges from 0 (F) to 4 (A), the latter being the "best" score. See https://en.wikipedia.org/wiki/Academic_grading_in_the_United_States
# 
# Class rank is determined by comparing your GPA to the GPA of people in the same grade. See https://blog.prepscholar.com/what-is-class-rank-why-is-it-important, but it seems this is not the kind of rank included in the dataset. 
# 
# The https://stats.idre.ucla.edu/stata/dae/logistic-regression/ states
# 
# "
# The variable rank takes on the values 1 through 4. Institutions with a rank of 1 have the highest prestige, while those with a rank of 4 have the lowest.
# "
# 
# so this is not really a "Class rank" but an "Institution rank".

# ## 1.1 Attempt to obtain deterministic results

# In[32]:


# https://github.com/pytorch/pytorch/issues/11278
def seed_everything(seed=1234):
    import random
    import numpy as np
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything()


# ## 1.2 Load the data
# To load the data and format it nicely, we will use two very useful packages called Pandas and Numpy. You can read on the documentation here:
# - https://pandas.pydata.org/pandas-docs/stable/
# - https://docs.scipy.org/

# In[33]:


# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('student_data.csv')

# Printing out the first 10 rows of our data
data[:10]


# In[34]:


data.describe()


# ## 1.3 Plot the data
# 
# First let's make a plot of our data to see how it looks. In order to have a 2D plot, let's ingore the rank.

# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


# Importing matplotlib
import matplotlib.pyplot as plt

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted],
                s=25, color='cyan', edgecolor='k', label='admitted')    
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected],
                s=25, color='red', edgecolor='k', label='rejected')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    plt.legend()
    
# Plotting the points
plot_points(data)
plt.show()


# The data is not as nicely separable as we hoped it would. Maybe it would help to take the rank into account? Let's make 4 plots, each one for each rank.

# In[37]:


# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()


# This looks more promising, as it seems that the lower the rank, the higher the acceptance rate. Let's use the rank as one of our inputs.

# ## 1.4 Scale the data
# The next step is to scale the data. The range for grades is 1.0-4.0, whereas the range for test scores is 200-800, which is much larger. Let's transform both features into the range 0-1.

# In[38]:


# TODO: Scale the columns
data['gpa'] = ((data['gpa'] - 1.0) / (4.0 - 1.0))
data['gre'] = ((data['gre'] - 200.0) / (800.0 - 200.0))

# Printing the first 10 rows of our processed data
data[:10]


# In[39]:


data.describe()


# ## 1.5 One-hot encode the rank
# 
# The rank variable naturally seems categorical: a school ranked '1' is not 4 times more prestigious than a school ranked '4'.
# 
# This dataset is too small to see any negative consequences of leaving this variable as numerical (try this on your own!),
# but let's use the `get_dummies` function in Pandas in order to one-hot encode the rank variable.

# In[40]:


# Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop(['rank'], axis=1)

# Print the first 10 rows of our data
one_hot_data[:10]


# In[41]:


one_hot_data.describe()


# In[42]:


print(one_hot_data['gpa'].min(), one_hot_data['gpa'].max())
print(one_hot_data['gre'].min(), one_hot_data['gre'].max())


# ## 1.6 Drop one of the one-hot-encoded rank variables
# 
# One of the columns is dropped in order to remove the multicollinearity introduced by one-hot-encoding.

# In[43]:


one_hot_data = one_hot_data.drop('rank_4', axis=1)


# In[44]:


one_hot_data.describe()


# ## 1.7 Split the data into Training and Testing

# In order to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% of the total data.

# In[45]:


sample = np.random.choice(one_hot_data.index,
                          size=int(len(one_hot_data)*0.7), replace=False)
train_data = one_hot_data.iloc[sample]
test_data = one_hot_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of testing samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])


# ## 1.8 Split the data into features and targets (labels)
# Now, as a final step before the training, we'll split the data into features (X) and targets (y).

# In[46]:


features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])


# ## 1.9 Fit logistic regression model using train data

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[48]:


model = LogisticRegression(random_state=0, penalty='none', solver='lbfgs')


# In[49]:


model.fit(features, targets)


# ## 1.10 Calculate performance indicators on the test data

# In[50]:


# Confusion matrix whose i-th row and j-th column entry indicates the number of
# samples with true label being i-th class and predicted label being j-th class.

np.array([['TN', 'FP'],
          ['FN', 'TP']])


# In[51]:


confusion_matrix(targets_test, model.predict(features_test))


# In[52]:


print(classification_report(targets_test, model.predict(features_test)))


# In[53]:


print(accuracy_score(targets_test, model.predict(features_test)))


# ## 1.11 Save the model

# In[54]:


import joblib


# In[55]:


joblib.dump(model, 'model.joblib')


# ## 1.12 Test prediction using the saved model

# In[56]:


model_local = joblib.load('model.joblib')


# Predict an admission of a student with a high gre, gpa and coming from the most prestigious school.

# In[57]:


model_local.predict([[0.8, 0.8, 1, 0, 0]])


# Predict an admission of a student with a low gre, gpa and coming from the least prestigious school.

# In[58]:


model_local.predict([[0.2, 0.2, 0, 0, 0]])


# # 2. Deploy the model to AKS (Azure Kubernetes Service)

# This part of the notebook is based on https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training/train-on-local and https://github.com/solliancenet/udacity-intro-to-ml-labs/tree/master/aml-visual-interface/lab-22/notebook

# ## 2.1 Install dependencies

# Obtain the Python version running inside of this notebook

# In[59]:


import sys
print(sys.version_info)
PY_VERSION = ".".join([str(x) for x in sys.version_info[0:2]])
print(PY_VERSION)


# Add the pip installation path to Python's sys.path

# In[60]:


import os
HOME = os.environ['HOME']
sys.path.insert(0, f"{HOME}/.local/lib/python{PY_VERSION}/site-packages")
print(sys.path)


# Add binaries installed by pip user mode to PATH

# In[61]:


os.environ["PATH"] += os.pathsep + f"{HOME}/.local/bin"


# ## 2.1 Install azure-ml Python SDK

# In[62]:


get_ipython().system('pip install --user azureml.core==1.12.0.post1')


# In[63]:


get_ipython().system('pip show azureml.core')


# In[64]:


import azureml.core
from azureml.core import Environment
from azureml.core import Experiment
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core.webservice import Webservice, AksWebservice

print("Azure ML SDK version:", azureml.core.VERSION)


# ## 2.2 Install azure cli

# In[65]:


get_ipython().system('pip install --user azure-cli==2.11.0')


# In[66]:


get_ipython().system('pip show azure-cli')


# ## 2.3 Add az ml cli extension

# In[67]:


get_ipython().system('az extension add -n azure-cli-ml')


# In[68]:


get_ipython().system('az ml -h')


# ## 2.4 Initialize workspace

# The resource group 'aks-intro' is assumed to already exist

# In[69]:


project = 'aks-intro'


# Import the Azure subscription ID from the local file. It is not very confidential, but importing from file is used here to demonstrate this method of handling sensitive data in Jupyter notebooks. When running this notebook yourself please modify the `ws.py` file to contain your subscription ID.

# In[70]:


from ws import subscription_id


# In[71]:


ws = Workspace.create(name=project,
                      subscription_id=subscription_id,
                      resource_group=project, 
                      location='West US',
                      exist_ok=True)
ws.write_config()


# ## 2.5 Create an experiment in the workspace

# In[72]:


experiment = Experiment(workspace=ws, name=project)


# ## 2.6 Register the model in the workspace

# In[73]:


model_azure = Model.register(description="Student admissions logistic regression model",
                             model_name=project,
                             model_path='model.joblib',
                             tags={'project': project},
                             workspace=ws)


# ## 2.7 Create the inference script

# The cell below is taken from https://github.com/solliancenet/udacity-intro-to-ml-labs/tree/master/aml-visual-interface/lab-22/notebook.
# 
# In order to deploy an inference endpoint on a AKS cluster one needs several pieces:
# - a model in a serialized form to be stored and read from disk - taken care by `joblib`
# - a Python inference script - written in the cell below
# - a Python environment for running the script in - taken care by packaging the environment as a Docker image
# - a webservice component deployed on AKS that allows for calling the inference script by issuing HTTP requests - taken care by an Nginx webserver running in the containers and exposed to the world by the Kubernetes `NodePort` service and a loadbalancer.

# The inference script below loads the serialized model in the `init` method and returns a prediction for the provided json input in the `run` method.
# 
# The availability of the Python imports, on the top of the script, are taken care by a Docker image created according to the specified coda environment configuration (see the cell below the next one).

# In[74]:


get_ipython().run_cell_magic('writefile', 'inference.py', '\nimport json\nimport time\n\nimport numpy as np\nimport pandas as pd\nimport azureml.core\nfrom azureml.core.model import Model\nimport joblib\n\ncolumns = [\'gre\', \'gpa\', \'rank_1\', \'rank_2\', \'rank_3\']\n\ndef init():\n    global model\n    \n    print("Azure ML SDK version:", azureml.core.VERSION)\n    model_name = \'aks-intro\'\n    print(\'Looking for model path for model: \', model_name)\n    model_path = Model.get_model_path(model_name=model_name)\n    print(\'Looking for model in: \', model_path)\n    model = joblib.load(model_path)\n    print(\'Model initialized:\', time.strftime(\'%H:%M:%S\'))\n\ndef run(input_json):     \n    try:\n        inputs = json.loads(input_json)\n        data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)),\n                               columns = columns)\n        # Get the predictions...\n        prediction = model.predict(data_df)\n        prediction = json.dumps(prediction.tolist())\n    except Exception as e:\n        prediction = str(e)\n    return prediction')


# ## 2.8 Define the conda environment for the inference script

# In[75]:


get_ipython().system('cat ./environment.yml')


# In[76]:


environment = Environment.from_conda_specification(project, './environment.yml')
environment.register(workspace=ws)


# ## 2.9 Define the Docker image configuration

# Use the provided inference script and configure the conda environment in the image as specified

# In[77]:


inference_config = InferenceConfig(entry_script='inference.py', environment=environment)


# ## 2.10 Define the configuration of the inference container

# Give the single "replica" container running on AKS 0.2 GB of RAM and 0.1 CPU core. Do not scale the number of containers depending on the load. For more configuration options see  https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice.akswebservice

# In[78]:


aks_config = AksWebservice.deploy_configuration(
    autoscale_enabled=False,
    cpu_cores=0.1,
    description='Student admissions logistic regression model',
    memory_gb=0.2,
    num_replicas=1,
    tags={'project': project})


# ## 2.11 Use an existing AKS cluster as the deployment target

# In[79]:


attach_config = AksCompute.attach_configuration(cluster_name=project,
                                                cluster_purpose='DevTest',  # allows 1 node
                                                resource_group=project)


# The cell below attaches the existing Kubernetes cluster as a compute target in the Azure ML workspace. It may take about 5 minutes.

# In[80]:


aks_target = ComputeTarget.attach(attach_configuration=attach_config,
                                  name=project,  # limit of 16 characters
                                  workspace=ws)
aks_target.wait_for_completion(True)


# ## 2.12 Deploy the model and inference script to AKS

# Up to this point the compute target Kubernetes cluster is empty. The step below will create several resources on the cluster, including the pod serving the prediction endpoint.
# 
# The creation of all the resources needed by the prediction endpoint deployed on the Kubernetes cluster may take about 10 minutes.

# In[83]:


aks_service = Model.deploy(deployment_config=aks_config,
                           deployment_target=aks_target,                           
                           inference_config=inference_config,
                           models=[Model(ws, name=project)],  # take the model from ws
                           name=project,
                           overwrite=True,
                           workspace=ws)


# In[82]:


aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)


# In[84]:


print(aks_service.get_logs())


# ## 2.13 Test the inference on a batch of two data points

# In[85]:


import json

batch = [[0.8, 0.8, 1, 0, 0], 
         [0.2, 0.2, 0, 0, 0]]

batch_json = json.dumps(batch)

result = aks_service.run(batch_json)
print('Predictions for batch', result)


# ## 2.14 Test the inference using the webservice running on AKS

# Save the primary api key for endpoint authentication into the `primary_key.py` file

# In[86]:


get_ipython().system('az login > /dev/null')


# In[87]:


get_ipython().run_cell_magic('bash', '-s "$project"', 'project=$1\nprimary_key=$(az ml service get-keys --name $project \\\n            | python -c \'import json,sys;print(json.load(sys.stdin)["primaryKey"])\')\necho "api_key=\\"$primary_key\\"" > primary_key.py')


# Import the api key from file

# In[88]:


from primary_key import api_key


# In[89]:


import requests

url = aks_service.scoring_uri
print('AKS Service: {} scoring URI is: {}'.format(project, url))
headers = {'Content-Type': 'application/json',
          'Authorization': f'Bearer {api_key}'}


# In[90]:


response = requests.post(url, batch_json, headers=headers)
print('Predictions for batch')
print(response.text)


# Test the inference on the command line using curl. make the payload spaces-free.

# In[91]:


batch_json


# In[92]:


batch_json_wo_spaces = json.dumps(batch, separators=(',', ':'))


# In[93]:


batch_json_wo_spaces


# In[116]:


get_ipython().system('echo curl -X POST            -H "\\"Content-Type: application/json\\""            -H "\\"Authorization: Bearer $api_key\\""            --data "\\"$batch_json_wo_spaces\\"" $url 2> /dev/null')


# In[95]:


get_ipython().system('curl -X POST       -H \'Content-Type: application/json\'       -H "Authorization: Bearer $api_key"       --data "$batch_json_wo_spaces" $url 2> /dev/null')


# # 3. Explore the resources created on AKS

# Please consult this post for more details https://liupeirong.github.io/amlKubernetesDeployment/

# Verify access to the Kubernetes cluster

# In[96]:


get_ipython().system('kubectl version')


# ## 3.1 Explore the inference container running on AKS

# The AKS cluster is running a single worker

# In[97]:


get_ipython().system('kubectl get node')


# The container is in the $project (azureml-aks-intro) namespace

# In[98]:


get_ipython().system('kubectl get namespaces')


# The pod serving the prediction endpoint is part of the "aks-intro" deployment.

# In[99]:


get_ipython().system('kubectl -n azureml-aks-intro get pod')


# In[100]:


get_ipython().system('kubectl -n azureml-aks-intro get deployment')


# The model and inference script are available under `/var/azureml-app`.

# In[101]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'ls -al /var/azureml-app'")


# In[102]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'ls -al /var/azureml-app/azureml-models/aks-intro/1/'")


# Below, our `inference.py` script can be recognized.

# In[103]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'cat /var/azureml-app/inference.py'")


# The container is running an Nginx webserver on port 5001 as a reverse proxy for Gunicorn running on a port number over 30000, serving the HTTP request by the Flask app.

# In[104]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'ps aux'")


# Let's install the `netstat` toool to list the ports opened in the prediction endpoint container.

# In[105]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'apt-get update'")


# In[106]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'apt-get install -y net-tools'")


# In[107]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'netstat -ntlp'")


# In[108]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'curl localhost:5001'")


# In[109]:


get_ipython().system('kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath=\'{.items[0].metadata.name}\')          -- bash -c \'grep -E "import flask|Healthy" /var/azureml-server/app.py\'')


# In[110]:


get_ipython().system("kubectl -n azureml-aks-intro exec -it          $(kubectl -n azureml-aks-intro get pod          -lazuremlappname=aks-intro -o jsonpath='{.items[0].metadata.name}')          -- bash -c 'cat /etc/nginx/sites-available/app'")


# The Nginx webserver port is exposed to the world using the Kubernetes `NodePort` service and placed behind a loadbalancer managed by the `azureml-fe` service in the default namespace.

# In[111]:


get_ipython().system('kubectl -n azureml-aks-intro get service aks-intro')


# In[112]:


get_ipython().system('kubectl -n default get service')


# # 4. Cleanup the Azure ML project workspace

# Note that the cleanup below is incomplete, and a complete cleanup would need to happen on a lower level API.

# In[117]:


aks_service.delete()


# In[118]:


aks_target.detach()


# In[119]:


model_azure.delete()


# In[120]:


ws.delete()


# Note also that the Kubernetes cluster is not removed by the above cleanup.
