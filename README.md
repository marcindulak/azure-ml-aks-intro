# Deploy a locally trained model to Azure ML (Machine Learning) AKS (Azure Kubernetes Service)

This tutorial creates a https://scikit-learn.org/ logistic regression model, using a locally running jupyter notebook instance.
https://www.pulumi.com/ is used to configure an instance of AKS where the model is deployed for predictions.

The instructions can be performed using the Azure free account https://azure.microsoft.com/en-us/free/
and the setup commands are provided for Ubuntu Linux 18.04.

# Setup software dependencies

1.1. Install Docker https://docs.docker.com/engine/install/

1.2. Install https://direnv.net/ and populate .envrc
   ```sh
   cat .envrc
   export PATH=$PWD/bin:$PATH
   export PATH=$PWD/bin/pulumi:$PATH
   export OS=linux
   export PULUMI_CLI_VERSION=v2.9.0
   export KUBECTL_VERSION=v1.18.8
   export KUBECONFIG=$PWD/kubeconfig.yaml
   direnv allow
   ```

1.3. Install nodejs, e.g. on Ubuntu
   ```sh
   sudo apt update
   sudo apt -y install curl dirmngr apt-transport-https lsb-release ca-certificates
   curl -sL https://deb.nodesource.com/setup_12.x | sudo bash
   sudo apt-get -y install nodejs
   ```

1.4. Install pulumi-cli https://www.pulumi.com/docs/get-started/install/
   ```sh
   curl -sLO https://get.pulumi.com/releases/sdk/pulumi-${PULUMI_CLI_VERSION}-linux-x64.tar.gz
   ```
   ```sh
   mkdir -p bin
   tar zxf pulumi-${PULUMI_CLI_VERSION}*x64.tar.gz
   mv pulumi bin/pulumi-${PULUMI_CLI_VERSION}
   ln -s pulumi-${PULUMI_CLI_VERSION} bin/pulumi
   pulumi version | grep ${PULUMI_CLI_VERSION}
   ```

1.5. Install kubectl, the Kubernetes command-line tool https://kubernetes.io/docs/tasks/tools/install-kubectl/
   ```sh
   curl -Lo kubectl-${KUBECTL_VERSION} https://storage.googleapis.com/kubernetes-release/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl
   chmod u+x kubectl-${KUBECTL_VERSION}
   mv kubectl-${KUBECTL_VERSION} bin
   ln -s kubectl-${KUBECTL_VERSION} bin/kubectl
   kubectl version --client | grep ${KUBECTL_VERSION}
   ```

1.6. Configure kubectl tab completion, e.g. for bash
   ```sh
   echo 'source <(kubectl completion bash)' >>~/.bashrc
   ```

1.7. Create a new pulumi project (if not already present)
   ```sh
   mkdir pulumi
   cd pulumi
   pulumi new azure-typescript --name azure-ml-aks-intro
   ```

1.8. Install npm dependencies (if creating a new pulumi project)
   ```sh
   npm install @pulumi/azure
   npm install @pulumi/azuread
   npm install @pulumi/kubernetes
   npm install @pulumi/random
   ```
   If the pulumi project already exist just install the defined npm dependencies
   ```sh
   npm install
   ```

1.9. Install https://docs.microsoft.com/en-us/cli/azure/install-azure-cli, e.g. on Ubuntu
   ```sh
   sudo apt-get update
   sudo apt-get install ca-certificates curl apt-transport-https lsb-release gnupg gnupg2
   curl -sL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null
   AZ_REPO=$(lsb_release -cs)
   echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $AZ_REPO main" | sudo tee /etc/apt/sources.list.d/azure-cli.list
   sudo apt-get update
   sudo apt-get install -y azure-cli
   ```
     
# Create Kubernetes cluster with pulumi

2.1. Login to Azure using azure cli
   ```sh
   az login > /dev/null
   ```

2.2. Create the kubernetes cluster in Azure using pulumi
   ```sh
   pulumi stack ls
   pulumi stack select dev
   pulumi up
   ```
   The cluster consists of a minimal configuration one Standard_B2s worker with 2 vCPUs and 4GiB or RAM.
   The pulumi code is based on the repository associated with the talk
   "Unleash the power of containers with Pulumi and AKS" https://www.youtube.com/watch?v=j4W5XHCRi74

2.3. Save kubeconfig file and test access to the cluster
    ```sh
    pulumi stack output kubeConfig > ../kubeconfig.yaml
    kubectl get node | grep Ready
    ```

# Train and deploy the model to AKS

3.1. Start Jupyter server with Docker, and follow the [aks-intro.ipynb](jupyter/aks-intro.ipynb) notebook
   ```sh
   cd ..
   docker-compose -f docker-compose.jupyter.yml up
   ```
   One could also use for example conda instead instead of Docker.

# Cleanup

4.1. Stop the jupyter server
   ```sh
   docker-compose -f docker-compose.jupyter.yml down
   ```

4.2. Remove the Kubernetes cluster
   ```sh
   cd pulumi
   pulumi destroy --yes
   ```
