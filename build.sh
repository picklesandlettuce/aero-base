#!/bin/bash
set -e


# Check if git repo is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Git repository has uncommitted changes. Please commit all changes before building."
    exit 1
fi

GIT_HASH=$(git rev-parse --short HEAD)

# Define the build stages in order
STAGES=(
    "0_basics.Dockerfile:aero-basics"
    "1_cuda.Dockerfile:aero-cuda"
    "2_toolchain.Dockerfile:aero-toolchain"
    "3_multimedia.Dockerfile:aero-multimedia"
    "4_ml.Dockerfile:aero-ml"
    "5_ros.Dockerfile:aero-ros"
    "6_apps.Dockerfile:aero-apps"
)

# Build each stage
PREV_TAG=""
for stage in "${STAGES[@]}"; do
    # Split stage on ':' - format expected: "dockerfile:name"
    dockerfile="${stage%:*}"  # Everything before the last ':'
    name="${stage##*:}"       # Everything after the last ':'
    TAG="$name:$GIT_HASH"

    # Check if image already exists
    if docker image inspect "$TAG" >/dev/null 2>&1; then
        echo "Image $TAG already exists, skipping"
        PREV_TAG="$TAG"
        continue
    fi

    echo "Building $TAG"

    if [ -n "$PREV_TAG" ]; then
        docker build --progress=plain -f "$dockerfile" --build-arg BASE_IMAGE="$PREV_TAG" -t "$TAG" .
    else
        docker build --progress=plain -f "$dockerfile" -t "$TAG" .
    fi

    PREV_TAG="$TAG"
    echo "Built $TAG"
    echo ""
done

# Tag final image as latest
FINAL_TAG="aero-apps:$GIT_HASH"
docker tag "$FINAL_TAG" "aero-base:$GIT_HASH"

echo "Build completed!"
echo "Final image: $FINAL_TAG"
echo "Also tagged as: aero-base:latest"