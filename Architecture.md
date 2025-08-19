```mermaid
flowchart TD

    subgraph UserSide["Local Environment"]
        User([User])
        CLI[CLI]
        Session[Python BedrockSession]
        FS[(Local File System)]
    end

    subgraph CloudSide["AWS Cloud"]
        S3[S3 Bucket]
        Bedrock[AWS Bedrock Service]
        Model[LLM Hosted in Bedrock]
    end

    User <--> CLI
    CLI <--> Session

    Session <--> |Read/Write| FS

    Session --> |Write| S3

    Session -->|API: Sends prompt| Bedrock
    Bedrock -->|API: Returns response| Session

    Bedrock --> Model
    Model --> Bedrock


```