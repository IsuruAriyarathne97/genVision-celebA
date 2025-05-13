import papermill as pm
from datetime import datetime

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# pm.execute_notebook(
#     'celebA-vAE.ipynb',  # Path to the input notebook
#     f'scripts/celebA-vAE-{timestamp}.ipynb'        # Path to the output notebook
# )

# pm.execute_notebook(
#     'celebA-vqVAE.ipynb',  # Path to the input notebook
#     f'scripts/celebA-vqVAE-{timestamp}.ipynb'        # Path to the output notebook
# )

pm.execute_notebook(
    'celebA-GAN.ipynb',  # Path to the input notebook
    f'scripts/celebA-GAN-{timestamp}.ipynb'        # Path to the output notebook
)

# pm.execute_notebook(
#     'celebA-BiGAN.ipynb',  # Path to the input notebook
#     f'scripts/celebA-BiGAN-{timestamp}.ipynb'        # Path to the output notebook
# )


# pm.execute_notebook(
#     'celebA-NGAN.ipynb',  # Path to the input notebook
#     f'scripts/celebA-NGAN-{timestamp}.ipynb'        # Path to the output notebook
# )

# pm.execute_notebook(
#     'test.ipynb',  # Path to the input notebook
#     f'scripts/test-{timestamp}.ipynb'        # Path to the output notebook
# )
