from pytorch_lightning import Trainer
from lightning import DumbSystem

model = DumbSystem()

trainer = Trainer(max_nb_epochs=3, gpus=1, early_stop_callback=None)
trainer.fit(model) 
