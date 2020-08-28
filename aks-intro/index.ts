import * as azure from "@pulumi/azure";
import * as azuread from "@pulumi/azuread";
import * as k8s from "@pulumi/kubernetes";
import * as random from "@pulumi/random";

const project = 'aks-intro';

const resourceGroup = new azure.core.ResourceGroup("resourceGroup", {
    name: project,
    location: 'West US',
    tags: {
        "themd:project": project,
    },
});

const password = new random.RandomPassword("password", {
    length: 20,
    special: true,
}).result;

const adApp = new azuread.Application("adApp", {
    name: project,
});
const adAppSP = new azuread.ServicePrincipal("adAppSP", {
    applicationId: adApp.applicationId,
});
const adAppServicePrincipalPassword = new azuread.ServicePrincipalPassword(project, {
    endDate: "2099-01-01T00:00:00Z",
    servicePrincipalId: adAppSP.id,
    value: password,
});

const kubernetesCluster = new azure.containerservice.KubernetesCluster("kubernetesCluster", {
    defaultNodePool: {
        name: "nodepool",
        nodeCount: 1,
        vmSize: "Standard_B2s",
        osDiskSizeGb: 30,
    },
    dnsPrefix: project,
    kubernetesVersion: "1.16.13",
    name: project,
    networkProfile: { networkPlugin: "azure" },
    resourceGroupName: resourceGroup.name,
    roleBasedAccessControl: { enabled: true },
    servicePrincipal: {
        clientId: adApp.applicationId,
        clientSecret: adAppServicePrincipalPassword.value,
    },
    tags: {
        "themd:project": project,
    },
});

const provider = new k8s.Provider("provider", {
    kubeconfig: kubernetesCluster.kubeConfigRaw
})

export let clusterName = kubernetesCluster.name;
export let kubeConfig = kubernetesCluster.kubeConfigRaw;
