"""
One-time setup: generates credentials.yaml with bcrypt-hashed passwords.

Default accounts created:
  admin / Admin@123  (role: admin)
  user  / User@123   (role: user)

To add more users or change passwords, edit DEFAULT_USERS below and re-run.

IMPORTANT: credentials.yaml is gitignored.
For production (Streamlit Cloud), store user data in Streamlit Secrets instead.
"""
import bcrypt
import yaml

DEFAULT_USERS = {
    "admin": {
        "plain_password": "Admin@123",
        "name": "Administrator",
        "role": "admin",
    },
    "user": {
        "plain_password": "User@123",
        "name": "Regular User",
        "role": "user",
    },
}


def main():
    config = {"users": {}}

    for username, data in DEFAULT_USERS.items():
        hashed = bcrypt.hashpw(
            data["plain_password"].encode(), bcrypt.gensalt()
        ).decode()
        config["users"][username] = {
            "name": data["name"],
            "role": data["role"],
            "password_hash": hashed,
        }

    with open("credentials.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("credentials.yaml created successfully.\n")
    print("Default login credentials:")
    print("-" * 35)
    for username, data in DEFAULT_USERS.items():
        print(
            f"  Username : {username}\n"
            f"  Password : {data['plain_password']}\n"
            f"  Role     : {data['role']}\n"
        )
    print("IMPORTANT: Change these passwords before deploying to production!")


if __name__ == "__main__":
    main()
