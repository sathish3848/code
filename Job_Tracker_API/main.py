from fastapi import FastAPI, Body, Form, File, UploadFile, HTTPException

app = FastAPI()

applications = []


@app.get("/")
def home():
    return {"message": "hi i am sathish, here to create the fast api projict"}


@app.post("/applications")
def apply_job(
    company: str = Body(...), role: str = Body(...), status: str = Body("Applied")
):

    count = len(applications)
    new_id = count + 1

    application = {
        "id": new_id,
        "company": company,
        "role": role,
        "status": status,
        "resume": None,
    }

    applications.append(application)
    return application


@app.post("/applications/form")
def apply_job_form(
    company: str = Form(...), role: str = Form(...), status: str = Form("Applied")
):
    count = len(applications)
    new_id = count + 1

    application = {
        "id": new_id,
        "company": company,
        "role": role,
        "status": status,
        "resume": None,
    }

    applications.append(application)
    return application


from pathlib import Path

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/applications/{application_id}/resume")
async def upload_resume(application_id: int, file: UploadFile = File(...)):
    application = None
    for item in applications:
        if item["id"] == application_id:
            application = item
            break

    if application is None:
        raise HTTPException(status_code=404, detail="Application not found")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()
    save_path = UPLOAD_DIR / f"resume_{application_id}.pdf"
    save_path.write_bytes(content)

    application["resume"] = save_path.name
    return application
