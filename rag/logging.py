def log(msg, cb=None):
    print(msg)
    if cb:
        cb(msg)
