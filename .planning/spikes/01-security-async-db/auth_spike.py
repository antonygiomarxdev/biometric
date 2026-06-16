import jwt
from datetime import datetime, timedelta, timezone
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

# 1. Configurar hashing con Argon2 (Estándar OWASP actual para biometría/alta seguridad)
# pwdlib reemplaza a passlib que está abandonado
password_hash = PasswordHash((Argon2Hasher(),))

def hash_password(password: str) -> str:
    return password_hash.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)

# 2. Configurar JWT con PyJWT (reemplaza a python-jose)
SECRET_KEY = "super-secret-key-for-spike-only-do-not-use-in-prod"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    
    # PyJWT usa jwt.encode directamente
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

if __name__ == "__main__":
    print("=== SPIKE: Seguridad Moderna ===")
    pwd = "secure_biometric_password_123!"
    print(f"Original: {pwd}")
    
    hashed = hash_password(pwd)
    print(f"Hashed (Argon2): {hashed}")
    
    is_valid = verify_password(pwd, hashed)
    print(f"Verificación: {'EXITOSA' if is_valid else 'FALLIDA'}")
    
    # Prueba JWT
    token_data = {"sub": "investigador_forense", "role": "admin"}
    token = create_access_token(token_data)
    print(f"\nJWT Generado:\n{token}")
    
    decoded = decode_token(token)
    print(f"JWT Decodificado: {decoded}")
