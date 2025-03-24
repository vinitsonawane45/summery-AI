import secrets

secret_key = secrets.token_hex(32)  # Generates a 64-character (32-byte) secret key
print(secret_key)
