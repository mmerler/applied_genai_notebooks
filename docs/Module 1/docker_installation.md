# Docker Installation (Optional)

Docker is a platform that allows developers to build, share, and run applications in isolated environments called **containers**. Containers make it easy to package applications with all their dependencies, ensuring they run reliably across different systems.

> **Note:** Docker is optional for this course. All notebooks can be run directly with Python and the required dependencies. Docker is only needed if you want to containerize your applications for deployment.

## Why Use Docker?
- **Portability:** Run your application anywhere without compatibility issues.
- **Efficiency:** Lightweight compared to traditional virtual machines.
- **Scalability:** Simplifies deploying and scaling applications.

---

## Installing Docker

Follow these steps to install Docker on your operating system:

### Windows
1. Download Docker Desktop for Windows from the [official Docker website](https://www.docker.com/products/docker-desktop).
2. Run the installer and follow the on-screen instructions.
3. After installation, restart your computer if prompted.
4. Open Docker Desktop and sign in or create a Docker Hub account.

### macOS
1. Download Docker Desktop for Mac from the [official Docker website](https://www.docker.com/products/docker-desktop).
2. Double-click the downloaded `.dmg` file and drag the Docker icon to your Applications folder.
3. Open Docker Desktop from your Applications folder.
4. Sign in or create a Docker Hub account.

### Linux

Visit the [Docker Linux installation guide](https://docs.docker.com/engine/install/) for detailed instructions for your distribution.

---

## Installing Docker Compose

Docker Compose is a tool for defining and running multi-container Docker applications. Follow these steps to install Docker Compose:

### Windows and macOS
Docker Compose is included with Docker Desktop. No additional installation is required.

### Linux
1. Download the current stable release of Docker Compose:
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/download/$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -oP '"tag_name": "\K[^"]+')/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   ```
2. Apply executable permissions to the binary:
   ```bash
   sudo chmod +x /usr/local/bin/docker-compose
   ```
3. Verify the installation:
   ```bash
   docker-compose --version
   ```

---

## Post-Installation Setup

### Verify Docker is Running
Run the following command to check if Docker is installed and running:
```bash
docker --version
```

### Test Docker
Run the following command to test Docker:
```bash
docker run hello-world
```
This will download a test image and run it in a container. If successful, you’ll see a "Hello from Docker!" message.

### Test Docker Compose
To test Docker Compose, create a `docker-compose.yml` file with the following content:
```yaml
services:
  hello-world:
    image: hello-world
```
Then run:
```bash
docker-compose up
```
If successful, you’ll see the "Hello from Docker!" message.

---

## Resources
- [Docker Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

Happy Dockering! 🚀

