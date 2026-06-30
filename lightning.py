import lightning as L
from lightning.app.components import PythonScript

app = L.LightningApp(
    PythonScript("train.py")
)
