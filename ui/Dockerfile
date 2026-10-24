# UI image: the Angular app built once and served as static files by nginx (Phase 7).
# Stage 1 builds with Node; stage 2 contains only nginx + the built files (no Node, small image).
# .NET analogy: like publishing a Blazor WebAssembly app and hosting the output on a web server.

FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npx ng build

# nginx-unprivileged: the official nginx build that runs as a normal user (uid 101) instead of
# root (security plan B4), listening on 8080 because ports below 1024 need root.
FROM nginxinc/nginx-unprivileged:1.31-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/ui/browser /usr/share/nginx/html
EXPOSE 8080
