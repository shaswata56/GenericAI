import hashlib

def get_hash(file_path: str) -> str:
    hash_func = hashlib.new('md5')
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):
            hash_func.update(chunk)
    digest = hash_func.hexdigest()
    print(file_path, digest)
    return digest