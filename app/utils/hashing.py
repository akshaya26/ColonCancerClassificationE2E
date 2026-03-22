import hashlib

def get_hash(data: bytes):
    return hashlib.md5(data).hexdigest() 