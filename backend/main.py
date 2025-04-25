from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import query, validate, annotate
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(validate.router, prefix="/validate", tags=["Validate"])
app.include_router(annotate.router, prefix="/annotate", tags=["Annotate"])
