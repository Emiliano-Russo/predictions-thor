# Usar una imagen base oficial de Python 3.11-slim
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Establecer variables de entorno para OpenBLAS
ENV OPENBLAS_NUM_THREADS=1

# Copiar los archivos de requerimientos
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto en el que la aplicación correrá
EXPOSE 5000

# Comando para correr la aplicación usando Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]




