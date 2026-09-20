# syntax=docker/dockerfile:1
ARG NODE_IMAGE=node:22-alpine
ARG NGINX_IMAGE=nginx:1.28-alpine

# ---------- Build static bundle ----------
FROM ${NODE_IMAGE} AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# Empty = same origin (requests go through nginx proxy below)
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
# nginx workers run as an unprivileged user: make the bundle world-readable
# regardless of file permissions in the build context (e.g. public/ copied as 0600)
RUN npm run build && chmod -R a+rX,go-w dist

# ---------- Serve with nginx ----------
FROM ${NGINX_IMAGE}
ENV BACKEND_URL=http://backend:8000 \
    CLIENT_MAX_BODY_SIZE=1024m \
    NGINX_RESOLVER=auto
# nginx image renders /etc/nginx/templates/*.template with envsubst on start;
# 16-resolver.envsh подбирает DNS до рендеринга шаблона (имя должно сортироваться
# между 15-local-resolvers.envsh и 20-envsubst-on-templates.sh).
# chmod обязателен: entrypoint образа молча пропускает файлы без бита исполнения,
# и тогда в конфиг попало бы NGINX_RESOLVER=auto, с которым nginx не стартует.
COPY docker-entrypoint.d/16-resolver.envsh /docker-entrypoint.d/16-resolver.envsh
RUN chmod +x /docker-entrypoint.d/16-resolver.envsh
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD wget -qO /dev/null http://127.0.0.1/ || exit 1
