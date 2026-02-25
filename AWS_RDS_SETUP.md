# AWS RDS PostgreSQL Setup for VibeLens

## Prerequisites
- AWS Account with $200 credits (or free tier eligible)
- Project completed through Step 2 (embeddings generated)

---

## Step 1: Create RDS PostgreSQL Instance

### **1.1 Go to AWS Console**
```
https://console.aws.amazon.com/rds/
```

### **1.2 Create Database**

Click **"Create database"**

**Configuration:**
```
Engine: PostgreSQL
Version: PostgreSQL 15.x (latest)
Templates: Free tier (or Dev/Test for better performance)

Settings:
  DB instance identifier: vibelens-db
  Master username: postgres
  Master password: [Create strong password, save it!]

Instance configuration:
  Free tier: db.t3.micro (1 vCPU, 1GB RAM)
  OR with credits: db.t3.small (2 vCPU, 2GB RAM) - Recommended

Storage:
  Storage type: General Purpose SSD (gp3)
  Allocated storage: 20 GB (free tier limit)
  
Connectivity:
  Public access: Yes (for development)
  VPC security group: Create new
  
Additional configuration:
  Initial database name: vibelens
  Backup retention: 7 days
  Enable automated backups: Yes
```

Click **"Create database"**

Wait 5-10 minutes for creation...

---

## Step 2: Configure Security Group

### **2.1 Get Your IP Address**
```bash
curl ifconfig.me
# Output: 123.45.67.89 (your public IP)
```

### **2.2 Update Security Group**
```
1. Go to RDS Console → Databases → vibelens-db
2. Click on VPC security group
3. Edit inbound rules
4. Add rule:
   Type: PostgreSQL
   Protocol: TCP
   Port: 5432
   Source: My IP (auto-fills your IP)
   OR: Custom: 123.45.67.89/32
5. Save rules
```

---

## Step 3: Install pgvector Extension

### **3.1 Connect to RDS**
```bash
# Get endpoint from RDS console
# Format: vibelens-db.xxxxxxxxxx.us-west-2.rds.amazonaws.com

# Connect via psql (install if needed: brew install postgresql)
psql -h vibelens-db.xxxxxxxxxx.us-west-2.rds.amazonaws.com \
     -U postgres \
     -d vibelens

# Enter password when prompted
```

### **3.2 Install pgvector**
```sql
-- In psql:
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify
\dx
-- Should show "vector" in the list

-- Exit
\q
```

**Note:** If pgvector is not available, you may need to:
1. Use PostgreSQL 15+ with pgvector support
2. Or request AWS Support to enable it
3. Or use Aurora PostgreSQL which has pgvector built-in

---

## Step 4: Update VibeLens Configuration

### **4.1 Update .env File**
```bash
cd ~/Documents/VibeCurators

# Copy template
cp .env.example .env

# Edit with your RDS details
nano .env
```

**Fill in:**
```bash
PG_HOST=vibelens-db.xxxxxxxxxx.us-west-2.rds.amazonaws.com
PG_PORT=5432
PG_DATABASE=vibelens
PG_USER=postgres
PG_PASSWORD=your-master-password
```

Save: Ctrl+X, Y, Enter

---

## Step 5: Load Data to RDS
```bash
# Activate environment
conda activate VibeLens

# Run PostgreSQL loader
python load_to_postgres.py
```

**Expected output:**
```
=== Connecting to PostgreSQL ===
Host: vibelens-db.xxxxxxxxxx.us-west-2.rds.amazonaws.com
Connection successful!

=== Creating Table ===
Table 'movies' created successfully!

=== Bulk Insert ===
Insert complete in 3.45s

=== Creating HNSW Index ===
Index created in 52.34s

Database Loading Complete!
```

---

## Step 6: Test Search
```bash
python search_movies.py
```

---

## Cost Management

### **Free Tier (12 months):**
```
✅ 750 hours/month db.t3.micro
✅ 20GB storage
✅ 20GB backup

Your usage:
- Runtime: ~2-3 hours/month (testing)
- Storage: ~500MB
- Cost: $0
```

### **After Free Tier:**
```
db.t3.micro: ~$15/month
db.t3.small: ~$30/month

With $200 credits:
- Can run for 6+ months
```

### **Stop Instance When Not Using:**
```
RDS Console → vibelens-db → Actions → Stop temporarily
(Auto-restarts after 7 days, but saves credits)
```

---

## Troubleshooting

### **Connection timeout:**
```
Error: could not connect to server

Fix:
1. Check security group allows your IP
2. Verify public access is enabled
3. Check endpoint URL is correct
```

### **pgvector not available:**
```
Error: extension "vector" does not exist

Fix:
Use PostgreSQL 15.2+ or Aurora PostgreSQL
Contact AWS Support to enable pgvector
```

### **Password authentication failed:**
```
Fix:
1. Verify password in .env matches RDS master password
2. Check username is "postgres" (default)
3. Try resetting master password in RDS console
```

---

## Cleanup After Project
```bash
# Delete RDS instance (stops billing)
RDS Console → vibelens-db → Actions → Delete

# Options:
☐ Create final snapshot: No (for course project)
☑ I acknowledge that automated backups will be deleted

Type: delete me
Click: Delete
```

---

## Benefits of AWS RDS

✅ Professional experience
✅ Managed service (no server maintenance)
✅ Automatic backups
✅ Easy scaling
✅ $200 credits cover entire project
✅ Resume: "Experience with AWS RDS PostgreSQL"

