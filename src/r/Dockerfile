FROM r-base:4.5.1
WORKDIR /app
COPY src/r/install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R
COPY src/r/main.R /app/main.R
ENTRYPOINT ["Rscript", "/app/main.R"]
CMD ["--data-dir", "/app/data", "--output-dir", "/app/output"]
