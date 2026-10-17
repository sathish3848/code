# uv add "fastapi[standard]",  uv add uvicorn
# #uv run uvicorn api(file name):app --reload

from fastapi import FastAPI, Body, Form, UploadFile, File
from pydantic import BaseModel

app = FastAPI()


class mb(BaseModel):
    name: str
    price: float
    tag: str = None


@app.post("/oo")
def p(item: mb):
    return {"type": "JSON", "name": item.name, "price": item.price, "tag": item.tag}


@app.get("/")
def read_root():
    return {"hello": "word"}


@app.get("/item/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "type": str(type(item_id))}


from enum import Enum


class modelname(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/user/{modelname}")
def read(modelname: modelname):
    return {"model": modelname, "message": f"running {modelname.value}"}


sk = []


@app.post("/sk")
def post(model: dict):
    sk.append(model)
    return model


@app.post("/write")
def letter_write(topic: str = Body(..., media_type="text/plain")):
    return {"type": "text", "content": topic, "length": len(topic)}


@app.post("/pass")
def login(username: str = Form(...), password: int = Form(...)):
    return {"username": username, "password": password}


@app.post("/folder")
def file_reader(file: UploadFile = File(...)):
    return {"type": "file", "name": file.filename, "content": file.content_type}


def greet(name):
    print("Hello, " + name)


def my_func(a, b, c):
    my_dict = {"name": "Sathish", "skills": ["Python", "FastAPI"]}
    numbers = [1, 2, 3, 4, 5]
    return my_dict
