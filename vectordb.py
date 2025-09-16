import torch
import os

class VectorDB:
    """The vector database data structure for storing documents contexts."""

    def __init__(self, path: str = "./database/") -> None:
        """The initialize function when the data type is created.
        If the data is already stored in ./database, then the context is loaded
        from the files.
        
        Parameters
        ----------
        path : str
            The path to the vector database, defaults to ./database
        
        Returns
        -------
        None
        """
        self.vectordb_path = path
        self.vec_data = {} # store the embedded vector
        self.vec_metadata = {} # store document id of a chunk
        if not os.path.exists(path):
            os.makedirs(path)
        elif os.path.exists(os.path.join(path, "vector.pt")) and os.path.exists(os.path.join(path, "metadata.pt")):
            self.load_data()

    def load_data(self) -> None:
        """Load the vector database from the 'vector.pt' and 'metadata.pt' files, if exists.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        data_path = os.path.join(self.vectordb_path, "vector.pt")
        metadata_path = os.path.join(self.vectordb_path, "metadata.pt")
        self.vec_data = torch.load(data_path)
        self.vec_metadata = torch.load(metadata_path)
    
    def add_vector(self, id: str, vector: torch.Tensor, metadata: str) -> None:
        """Add the embedded vector into 'vec_data' and 'vec_metadata'.
        
        Parameters
        ----------
        id : str
            The id (key) for access the vector in dictionary
        vector : torch.Tensor
            The embedded vector correspond to the id
        metadata : tuple
            The metadata contains the sources related to the text (document_id, url)
        
        Returns
        -------
        None
        """
        self.vec_data[id] = vector
        self.vec_metadata[id] = metadata
    
    def get_vector(self, id: str) -> torch.Tensor:
        """The helper function to get the vector with corresponding id
        
        Parameters
        ----------
        id : str
            The key to access the vector
            
        Returns
        -------
        torch.Tensor
            The vector correspond to the id
        """
        return self.vec_data.get(id)
    
    def get_metadata(self, id: str) -> str:
        """The helper function to get the metadata with corresponding id
        
        Parameters
        ----------
        id : str
            The key to access the vector
        
        Returns
        -------
        str
            The metadata (path) correspond to the id
        """
        return self.vec_metadata.get(id)
    
    def save_to_json(self) -> None:
        """Save the vector database to .json files, named as 'vector.pt' and 'metadata.pt'
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None, but the file will be saved in the destination at vectordb_path
        """
        data_path = os.path.join(self.vectordb_path, "vector.pt")
        metadata_path = os.path.join(self.vectordb_path, "metadata.pt")
        torch.save(self.vec_data, data_path)
        torch.save(self.vec_metadata, metadata_path)
    
    def get_topk_similar(self, Eq: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, list]:
        """The function to find the top k most similar context.
        
        Parameters
        ----------
        Eq : torch.Tensor
            The embedded query, which is inputted from user
        k : int
            The amount of top k most similar context, defaults to 5
        doc_id : int
            The document id to look up for (if any)
        
        Returns
        -------
        list
            The top k most similar context for the LLM to read
        """
        similarity = []
        stored_path = []
        for i, Ed in self.vec_data.items():
            # MaxSim
            scores = torch.matmul(Eq, Ed.T)
            max_row = scores.max(dim=1).values
            sim_result = max_row.sum()
            similarity.append(sim_result)
            stored_path.append(self.vec_metadata[i])
        
        result = torch.topk(torch.Tensor(similarity), k)
        return result.values, [stored_path[i] for i in result.indices]