class Recording:
    uuid: str
    voiced_file_path: str
    unvoiced_file_path: str
    voiced_file_path_flac: str
    unvoiced_file_path_flac: str
    text: str
    timestamp: str
    def __init__(self, uuid: str, voiced_file_path: str, unvoiced_file_path: str, voiced_file_path_flac: str = '', unvoiced_file_path_flac: str = '', text: str = '', timestamp: str = ''):
        self.uuid = uuid
        self.voiced_file_path = voiced_file_path
        self.unvoiced_file_path = unvoiced_file_path
        self.voiced_file_path_flac = voiced_file_path_flac
        self.unvoiced_file_path_flac = unvoiced_file_path_flac
        self.text = text
        self.timestamp = timestamp
            
    def __repr__(self):
        return f"Recording(name={self.name}, uuid={self.uuid}, voiced_file_path={self.voiced_file_path}, unvoiced_file_path={self.unvoiced_file_path})"
        
    def get_voiced_data(self):
        if self.voiced_file_path_flac == '':
            return Exception("Voiced FLAC file path is empty. This file needs to be converted.")
        if self.voiced_file_path == '':
            return Exception("Voiced file path is empty. There might be a bigger mistake waiting for you.")
        with open(self.voiced_file_path, 'rb') as f:
            return f.read()
    
    def get_unvoiced_data(self):
        if self.unvoiced_file_path_flac == '':
            return Exception("Unvoiced FLAC file path is empty. This file needs to be converted.")
        if self.unvoiced_file_path == '':
            return Exception("Unvoiced file path is empty. There might be a bigger mistake waiting for you.")
        with open(self.unvoiced_file_path_flac, 'rb') as f:
            return f.read()

    def to_dict(self) -> dict:
        dict = {
            "uuid": self.uuid,
            "voiced_file_path": self.voiced_file_path,
            "unvoiced_file_path": self.unvoiced_file_path,
            "voiced_file_path_flac": self.voiced_file_path_flac,
            "unvoiced_file_path_flac": self.unvoiced_file_path_flac,
            "text": self.text,
            "timestamp": self.timestamp
        }
        return dict

    def _save_voiced_to_file(self, data):
        return
    
    def _save_unvoiced_to_file(self, data):
        return
        
    def convert_to_flac(self, output_path: str):
        # Set Paths
        if self.voiced_file_path_flac == '':
            self.voiced_file_path_flac = output_path
        if self.unvoiced_file_path_flac == '':
            self.unvoiced_file_path_flac = output_path
