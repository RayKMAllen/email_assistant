```mermaid
flowchart TD

    subgraph UserSide["Local Environment"]
        User([User])
        CLI[CLI]
        Session[Python BedrockSession]
        FS[(Local File System)]
    end

    subgraph CloudSide["AWS Cloud"]
        Bedrock[AWS Bedrock API]
        Model[LLM Hosted in Bedrock]
    end

    User <--> CLI
    CLI <--> Session

    Session <--> |Read/Write| FS

    Session -->|Sends prompt| Bedrock
    Bedrock -->|Returns response| Session

    Bedrock --> Model
    Model --> Bedrock


```