pipeline {
    agent any

    environment {
        APP_NAME       = "ai-meeting-intelligence"
        DEST_IP        = "10.192.110.26"
        DEST_USER_NAME = "clstrdbgovin"
        DEPLOY_PATH    = "/home/clstrdbgovin/AI_Meeting_Intelligence_Platform_Production"

        HTTP_PROXY     = "http://192.0.2.12:8080"
        HTTPS_PROXY    = "http://192.0.2.12:8080"
        http_proxy     = "http://192.0.2.12:8080"
        https_proxy    = "http://192.0.2.12:8080"
        NO_PROXY       = "10.192.110.26,localhost,127.0.0.1"
        no_proxy       = "10.192.110.26,localhost,127.0.0.1"
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {

        stage('Generate Timestamp') {
            steps {
                script {
                    env.TIMESTAMP = sh(
                        script: "date +%Y%m%d_%H%M%S",
                        returnStdout: true
                    ).trim()
                    env.TAR_FILE = "/tmp/aimeet-backend_${env.TIMESTAMP}.tar.gz"
                    echo "Tar File: ${env.TAR_FILE}"
                }
            }
        }

        // The embedding weights are a Git LFS object. A checkout made without git-lfs holds a ~130-byte POINTER under the
        // same file name; it would be tarred, deployed, and the server would fail with "Error while deserializing header:
        // header too large" the first time a meeting is indexed. Fetch the real file here, or stop the build.
        stage('Verify Embedding Model') {
            steps {
                sh '''
cd "$WORKSPACE"
if command -v python3 >/dev/null 2>&1; then
    # git lfs pull if needed, then size + SHA-256 against the repository's LFS oid (standard library only).
    python3 -m backend.scripts.ensure_embedding_model || exit 1
    exit 0
fi
MODEL="$WORKSPACE/models/embeddings/all-MiniLM-L6-v2/model.safetensors"
SIZE=$(wc -c < "$MODEL" 2>/dev/null | tr -d ' ' || echo 0)
if [ "${SIZE:-0}" -lt 1000000 ]; then
    echo "model.safetensors is ${SIZE:-0} bytes (a Git LFS pointer) - running git lfs pull"
    git lfs install --local || true
    git lfs pull --include="models/embeddings/**" || true
    SIZE=$(wc -c < "$MODEL" 2>/dev/null | tr -d ' ' || echo 0)
fi
if [ "${SIZE:-0}" -lt 1000000 ]; then
    echo "ERROR: $MODEL is still only ${SIZE:-0} bytes, not the ~91 MB model."
    echo "Install git-lfs on the Jenkins agent (the model is fetched from the repository's own LFS remote)."
    exit 1
fi
echo "Embedding model weights OK (${SIZE} bytes)"
'''
            }
        }

        stage('Create Tar') {
            steps {
                // .env and .env.* files are excluded from the tarball intentionally —
                // the production .env is injected separately in 'Inject Production Env'
                // via a Jenkins Secret file credential (never stored in SCM).
                sh '''tar -czf "$TAR_FILE" -C "$WORKSPACE" --exclude=.git --exclude=.env --exclude=.env.* --exclude=venv --exclude=chroma_db --exclude=uploads --exclude=data .'''
            }
        }

        stage('Copy Tar To Server') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aimeet-deploy-ssh-password',
                    usernameVariable: 'SSH_USER',
                    passwordVariable: 'SSH_PASS'
                )]) {
                    sh '''sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no "$TAR_FILE" "$SSH_USER@$DEST_IP:/home/$DEST_USER_NAME/"'''
                }
            }
        }

        stage('Deploy On Server') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aimeet-deploy-ssh-password',
                    usernameVariable: 'SSH_USER',
                    passwordVariable: 'SSH_PASS'
                )]) {
                    sh '''sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$DEST_IP" "bash -s" <<EXTRA
set -e
TAR_FILE=/home/$DEST_USER_NAME/aimeet-backend_$TIMESTAMP.tar.gz
EXTRACT_DIR=/tmp/aimeet_backend_$TIMESTAMP
mkdir -p \$EXTRACT_DIR
tar -xzf \$TAR_FILE -C \$EXTRACT_DIR

BACKUP_DIR=/home/$DEST_USER_NAME/backup_backend_$TIMESTAMP
mkdir -p \$BACKUP_DIR
cp -rf $DEPLOY_PATH/backend \$BACKUP_DIR/ 2>/dev/null || true
cp -f $DEPLOY_PATH/requirements.txt \$BACKUP_DIR/ 2>/dev/null || true
cp -f $DEPLOY_PATH/run_backend.sh \$BACKUP_DIR/ 2>/dev/null || true

cp -rf \$EXTRACT_DIR/. $DEPLOY_PATH/
chmod +x $DEPLOY_PATH/run_backend.sh $DEPLOY_PATH/run.sh 2>/dev/null || true

rm -rf \$EXTRACT_DIR
rm -f \$TAR_FILE

cd $DEPLOY_PATH
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"
source venv/bin/activate
pip install -r requirements.txt
deactivate
echo "Deployment files in place and dependencies installed."
EXTRA'''
                }
            }
        }

        // ── Inject production .env BEFORE pip install and PM2 restart ──────────────
        // Uses a Jenkins 'Secret file' credential named 'aimeet-production-env'.
        // To create the credential: Jenkins → Manage Credentials → Add Credential
        //   Kind: Secret file  |  ID: aimeet-production-env  |  File: .env_server contents
        stage('Inject Production Env') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'aimeet-deploy-ssh-password',
                        usernameVariable: 'SSH_USER',
                        passwordVariable: 'SSH_PASS'
                    ),
                    file(
                        credentialsId: 'aimeet-production-env',
                        variable: 'PROD_ENV_FILE'
                    )
                ]) {
                    sh '''
                        sshpass -p "$SSH_PASS" scp -o StrictHostKeyChecking=no \
                            "$PROD_ENV_FILE" \
                            "$SSH_USER@$DEST_IP:$DEPLOY_PATH/.env"
                        echo "Production .env injected into $DEPLOY_PATH/.env"
                    '''
                }
            }
        }

        stage('Dependency vulnerability audit (never fails build)') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aimeet-deploy-ssh-password',
                    usernameVariable: 'SSH_USER',
                    passwordVariable: 'SSH_PASS'
                )]) {
                    sh '''sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$DEST_IP" "bash -s" <<EXTRA
set +e
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"
cd $DEPLOY_PATH
source venv/bin/activate

echo "===== Known vulnerabilities (pip-audit) ====="
if command -v pip-audit >/dev/null 2>&1 || pip install pip-audit >/dev/null 2>&1; then
    pip-audit || echo ">>> pip-audit found issues above. Deployment continues anyway."
else
    echo ">>> pip-audit could not be installed (check network/proxy). Skipping vulnerability scan."
fi

echo "===== Packages with newer versions (pip list --outdated) ====="
pip list --outdated || true

deactivate
echo "===== Audit complete. Proceeding with deployment. ====="
exit 0
EXTRA'''
                }
            }
        }

        stage('Reload via PM2') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aimeet-deploy-ssh-password',
                    usernameVariable: 'SSH_USER',
                    passwordVariable: 'SSH_PASS'
                )]) {
                    sh '''sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$DEST_IP" "bash -s" <<EXTRA
set -e
cd $DEPLOY_PATH
chmod +x run_backend.sh run.sh 2>/dev/null || true

LAUNCHER="run_backend.sh"
test -f "\$LAUNCHER" || LAUNCHER="run.sh"

pm2 delete $APP_NAME >/dev/null 2>&1 || true
pm2 start "\$LAUNCHER" --name $APP_NAME
pm2 save
echo "Deployment OK - $APP_NAME started via \$LAUNCHER."
EXTRA'''
                }
            }
        }

        // The PM2 stage only starts the process. Prove it came up: wait for /system/health, then run the deployment
        // preflight against the live server (PostgreSQL round trip, embedding model, FFmpeg, AIRAWAT LLM + Whisper;
        // ChromaDB is owned by the running API, so its round trip is skipped). No secrets are printed.
        stage('Post-Deploy Health') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aimeet-deploy-ssh-password',
                    usernameVariable: 'SSH_USER',
                    passwordVariable: 'SSH_PASS'
                )]) {
                    sh '''sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$DEST_IP" "bash -s" <<EXTRA
set +e
cd $DEPLOY_PATH
source venv/bin/activate
PORT=\$(python -c "from dotenv import dotenv_values; print(dotenv_values('.env').get('PORT') or 8000)" 2>/dev/null)
PORT=\${PORT:-8000}
echo "Waiting for the API on port \$PORT (up to 240 s)..."
OK=0
for i in \$(seq 1 48); do
    if curl -sf "http://127.0.0.1:\$PORT/api/v1/system/health" >/dev/null 2>&1; then OK=1; break; fi
    sleep 5
done
if [ "\$OK" != "1" ]; then
    echo "The API did not become healthy. Last PM2 log lines:"
    pm2 logs $APP_NAME --lines 60 --nostream
    exit 1
fi
bash healthcheck.sh "http://127.0.0.1:\$PORT/api/v1/system/health"
exit 0
EXTRA'''
                }
            }
        }

        stage('Post-Deploy Preflight') {
            steps {
                // A failing dependency (AIRAWAT outage, missing model, fallback database) marks the build UNSTABLE
                // instead of failed: the code is deployed and serving, but a human must look.
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                    withCredentials([usernamePassword(
                        credentialsId: 'aimeet-deploy-ssh-password',
                        usernameVariable: 'SSH_USER',
                        passwordVariable: 'SSH_PASS'
                    )]) {
                        sh '''sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$SSH_USER@$DEST_IP" "bash -s" <<EXTRA
set +e
cd $DEPLOY_PATH
source venv/bin/activate
python -m backend.scripts.preflight --skip-chromadb
exit \$?
EXTRA'''
                    }
                }
            }
        }

        stage('Cleanup Agent') {
            steps {
                sh 'rm -f "$TAR_FILE"'
            }
        }
    }

    post {
        always {
            cleanWs(deleteDirs: true)
        }
        success {
            echo "ai-meeting backend deployed via PM2 from OpenForge on ${DEST_IP}."
        }
        failure {
            echo "ai-meeting backend deployment FAILED - inspect logs:"
            echo "ssh ${DEST_USER_NAME}@${DEST_IP} pm2 logs ${APP_NAME} --lines 100"
        }
    }
}
