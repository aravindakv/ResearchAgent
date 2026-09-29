# 6. Cloud deployment (AWS)

> **Diagrams in this guide** (SVG files in `docs/images/`, linked relative to this file as `images/<name>.svg`; keep the `images/` folder next to the `.md` files):
> - `docs/images/aws-trust-zones.svg`: AWS deployment with trust zones

The local stack was designed to map onto managed services with small code changes. This guide describes the target and the order to build it in. Use `ap-south-1` (Mumbai) for low latency from India, and write everything as Terraform from the start so the environment is reproducible and can be destroyed when you're not using it.

## Service mapping and code changes

| Local | AWS | Code change |
|---|---|---|
| Caddy | Application Load Balancer + AWS WAF + ACM certificate | None |
| Compose services | ECS Fargate services, one per component | None |
| Docker networks | VPC with public, private, and isolated subnets; security groups | None |
| `pgvector/pgvector` | RDS for PostgreSQL 16 with the `vector` extension | None (connection string only) |
| Redis Streams | SQS standard queue (or ElastiCache Redis to keep the code as is) | Swap `XADD`/`XREADGROUP` for `send_message`/`receive_message`/`delete_message` |
| `reports` volume | S3 bucket with KMS encryption | Worker uploads; API returns a pre-signed URL |
| Compose secrets | Secrets Manager | Inject secrets into containers as environment variables via the task definition, or read them with the SDK |
| Static tokens | Cognito user pool | Replace `current_user()` with JWT validation |
| Logs | CloudWatch Logs | None (containers log to stdout) |

## Target network layout

![AWS deployment with trust zones](images/aws-trust-zones.svg)

*Diagram file: `docs/images/aws-trust-zones.svg` (linked here as `images/aws-trust-zones.svg`)*

A full run takes minutes, so the API never does the work inline: it enqueues a job and returns a job ID, a worker runs the graph, and the client polls or listens for progress. A minimal AWS setup:

| Concern | AWS choice |
|---|---|
| API and workers | ECS Fargate (or App Runner) |
| Queue | SQS (or Redis) |
| Database, vectors, checkpoints | RDS Postgres + pgvector |
| PDFs | S3 with pre-signed URLs |
| Secrets | Secrets Manager |
| MCP server | Its own container, over streamable HTTP |

Each service runs as its own ECS Fargate service with its own IAM task role and security group. Security groups allow only the arrows shown: load balancer to API, API to SQS, worker to MCP server, and worker to Postgres. The MCP server has no route to Postgres, S3, or Secrets Manager at all. Workers autoscale on SQS queue depth, and the API scales on CPU. S3, SQS, and Secrets Manager are reached through VPC endpoints, so that traffic never crosses the internet. In `ap-south-1`, a small setup (one API task, 1–3 workers, one MCP task, a `db.t4g.medium` RDS instance) is a reasonable starting size. Put per-user token budgets in place from day one.

In more detail:

The ALB sits in public subnets. The API, worker, and MCP services run in private subnets. RDS runs in isolated subnets with no internet route. Outbound traffic goes through a NAT gateway, optionally inspected by AWS Network Firewall with domain allowlists for the API and worker (OpenAI, LangSmith). S3, SQS, Secrets Manager, ECR, and CloudWatch are reached through VPC endpoints. Security groups allow only ALB → API, worker → MCP, and API/worker → RDS.

Each ECS service gets its own IAM task role. The API role can send to the queue and read `jobs`-related secrets. The worker role can consume the queue, write to `s3://reports/pdfs/*`, and read the OpenAI, TypeSafe, and LangSmith secrets. The MCP role can read only the Tavily secret. No task role has wildcard permissions.

## Build order

1. **Foundation.** Terraform state in an S3 bucket with locking. VPC module with three subnet tiers across two availability zones. VPC endpoints.
2. **Data.** RDS PostgreSQL 16 (start with `db.t4g.medium`), encrypted, `rds.force_ssl = 1`, automated backups. Run the schema and role scripts once through a temporary task or a bastion session. `CREATE EXTENSION vector;` is supported on current RDS PostgreSQL versions.
3. **Storage and queue.** S3 bucket with Block Public Access, SSE-KMS, and a lifecycle rule deleting PDFs after 30 days. SQS queue with a dead-letter queue after 3 receives. Set the visibility timeout above your job timeout.
4. **Secrets.** Secrets Manager entries for each key (OpenAI, Tavily, LangSmith, TypeSafe) and database password, with rotation for the database passwords. If you use Network Firewall allowlists, add the TypeSafe API host for the worker.
5. **Images.** ECR repositories with scan on push. A GitHub Actions workflow that builds, runs Trivy, pushes with the commit SHA as the tag, and deploys. Use GitHub OIDC to assume a deploy role instead of long-lived AWS keys.
6. **Services.** ECS cluster, task definitions with `readonlyRootFilesystem`, non-root user, and secrets from Secrets Manager. Service Connect or Cloud Map so the worker can reach `mcp` by name.
7. **Edge.** ALB with an HTTPS listener (ACM certificate), redirect from HTTP, and WAF with the AWS managed common rule set and a rate-based rule.
8. **Identity.** Cognito user pool and app client. API validates JWTs against the pool's JWKS.
9. **Scaling.** Worker autoscaling on the SQS `ApproximateNumberOfMessagesVisible` metric; API autoscaling on CPU.
10. **Observability.** CloudWatch dashboards and alarms for queue age, failed jobs, 5xx rate, and spend. CloudTrail and GuardDuty enabled for the account.

## Cost control

The fixed costs are the NAT gateway, the ALB, and RDS, which run whether or not you use the system. For a learning setup, run `terraform destroy` when you're done for the day, or scale ECS services to zero and stop RDS. Set AWS Budgets alerts, and keep spend limits on the OpenAI and Tavily projects, since LLM calls usually cost more than the infrastructure.

## Simpler alternative

Google Cloud Run with Cloud SQL (PostgreSQL with pgvector), Cloud Tasks or Pub/Sub, Cloud Storage, and Secret Manager has fewer moving parts than ECS and scales to zero, which suits a learning project. The same code changes apply.
