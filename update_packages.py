import pkg_resources
import subprocess

# Obtener la lista de librerías instaladas
installed_packages = pkg_resources.working_set

# Obtener los nombres de las librerías
packages_to_update = [package.project_name for package in installed_packages]

# Comando pip para actualizar cada librería
for package in packages_to_update:
    subprocess.call(['pip', 'install', '--upgrade', package])
