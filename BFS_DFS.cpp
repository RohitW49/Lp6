#include<iostream>
#include<vector>
#include<omp.h>
#include<queue>
using namespace std;
void parallelBFS(const vector<vector<int>>& graph,int startNode){
    vector<bool> visited(graph.size(),false);
    vector<int> traversal;
    queue<int>bfs ;
    bfs.push(startNode);
    visited[startNode]=true;
    while(!bfs.empty()){
        int curr=bfs.front();
        bfs.pop();
        #pragma omp parallel for
        {
            for(int neighbor:graph[curr]){
                #pragma omp critical
                if(!visited[neighbor]){
                    #pragma omp task
                    bfs.push(neighbor);
                    visited[neighbor]=true;
                }
            }
        }
        #pragma omp critical
        traversal.push_back(curr);
    }
    for(int node:traversal){
        cout<<node<<" ";
    }
    cout<<endl;
}

void dfsHelper(const vector<vector<int>>& graph,int curr , vector<bool>& visited,vector<int> & traversal){
    visited[curr]=true;
    traversal.push_back(curr);
    #pragma omp parallel for
    {
        for(int neighbor:graph[curr]){
            #pragma omp critical
            {
                if(!visited[neighbor]){
                    #pragma omp task
                    {
                        dfsHelper(graph,neighbor,visited,traversal);
                    }
                }
            }
        }
    }
}
void parallelDFS(const vector<vector<int>>& graph, int startNode){
    vector<bool> visited(graph.size(),false);
    vector<int> traversal;
    #pragma omp parallel
    #pragma omp single nowait
    {
        dfsHelper(graph,startNode,visited,traversal);
    }
    for(int node:traversal){
        cout<<node<<" ";
    }
    cout<<endl;
}


int main(){
    cout<<"Enter number of nodes :";
    int n;
    cin>>n;
    vector<vector<int>> graph(n);
    cout<<"Enter adjacency list for each node :\n";
    for(int i=0;i<n;++i){
        cout<<"Enters neighbors of node "<<i<<" in the end -1 : ";
        int neigh;
        while(cin>>neigh && neigh !=-1){
            graph[i].push_back(neigh);
        }
    }
    int startNode;
    cout<<"Enter start node :";
    cin>>startNode;
    cout<<"\nParallel BFS traversal : ";
    parallelBFS(graph,startNode);
    cout<<endl;
    cout<<"Parallel DFS traversal :";
    parallelDFS(graph,startNode);
    return 0;
}