import requests


def analyze_batch(files, api_url):
    """
    Send multiple images to the FastAPI backend
    and return their analysis results.
    """

    results = []

    for uploaded_file in files:

        try:
            uploaded_file.seek(0)

            response = requests.post(
                api_url,
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                },
                timeout=30
            )

            if response.status_code == 200:

                result = response.json()

                result["filename"] = uploaded_file.name

                results.append(result)

            else:

                results.append({
                    "filename": uploaded_file.name,
                    "error": (
                        f"Backend returned "
                        f"{response.status_code}"
                    )
                })

        except requests.exceptions.Timeout:

            results.append({
                "filename": uploaded_file.name,
                "error": "Request timed out"
            })

        except requests.exceptions.ConnectionError:

            results.append({
                "filename": uploaded_file.name,
                "error": "Could not connect to backend"
            })

        except Exception as e:

            results.append({
                "filename": uploaded_file.name,
                "error": str(e)
            })

    return results