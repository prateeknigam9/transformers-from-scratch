
import kagglehub

def fetch_data(url):
    path = kagglehub.dataset_download(url)
    return path
