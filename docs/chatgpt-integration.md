# ChatGPT Actions Integration

## Step 1: Create Custom GPT

1. Go to https://chat.openai.com
2. Profile → "My GPTs" → "Create a GPT"
3. Switch to "Configure" tab

**Name:** Chinvex Memory

**Instructions:**
```
You have access to the user's personal knowledge base via the Chinvex API.

CRITICAL RULES:
1. When the user asks about their projects, decisions, code, or past work, ALWAYS call getEvidence first.
2. If grounded=false, say "I couldn't find information about that in your memory." Do NOT make up an answer.
3. If grounded=true, synthesize an answer using ONLY the returned chunks. Cite sources.
4. Never claim to know something that isn't in the evidence pack.

When citing, use format: [source_uri:line_start-line_end]
```

## Step 2: Add Action

1. Scroll to "Actions"
2. Click "Create new action"
3. Click "Import from URL"
4. Enter: `https://chinvex.yourdomain.com/openapi.json`

## Step 3: Configure Authentication

1. Click "Authentication"
2. Select "API Key"
3. Set:
   - **Auth Type:** Bearer
   - **API Key:** Your `CHINVEX_API_TOKEN`

## Step 4: Test

Ask: "Search my Chinvex memory for hybrid retrieval"

Expected: GPT calls `/v1/evidence`, synthesizes answer

## Troubleshooting

### "Could not load schema"
- Verify: `curl https://chinvex.yourdomain.com/openapi.json`
- Check CORS in gateway.json

### "Authentication failed"
- Test: `curl -H "Authorization: Bearer $CHINVEX_API_TOKEN" https://chinvex.yourdomain.com/v1/contexts`
