# Per-Class Retrieval Recall@10

> Computed from cached embeddings; no new model runs.
> Labelled sentences: 4070 | Label instances: 5143

Spearman rs = -0.07 (nomic) and -0.16 (ATTACK-BERT): no meaningful relationship between class support and Recall@10 was found.

## Full Table (sorted by support, descending)

| Technique | Name | Nomic R@10 | Nomic Sup | AB R@10 | AB Sup |
| :--- | :--- | ---: | ---: | ---: | ---: |
| T1027 | Obfuscated Files or Information | 82.30% | 678 | 87.32% | 678 |
| T1140 | Deobfuscate/Decode Files or Information | 88.63% | 466 | 93.99% | 466 |
| T1059.003 | Windows Command Shell | 80.45% | 358 | 53.35% | 358 |
| T1055 | Process Injection | 66.19% | 278 | 82.37% | 278 |
| T1105 | Ingress Tool Transfer | 82.76% | 261 | 65.52% | 261 |
| T1106 | Native API | 70.53% | 207 | 54.11% | 207 |
| T1078 | Valid Accounts | 66.46% | 161 | 83.23% | 161 |
| T1090 | Proxy | 79.87% | 149 | 90.60% | 149 |
| T1071.001 | Web Protocols | 81.08% | 148 | 85.81% | 148 |
| T1082 | System Information Discovery | 61.79% | 123 | 81.30% | 123 |
| T1053.005 | Scheduled Task | 86.44% | 118 | 89.83% | 118 |
| T1003.001 | LSASS Memory | 94.64% | 112 | 85.71% | 112 |
| T1112 | Modify Registry | 85.05% | 107 | 90.65% | 107 |
| T1021.001 | Remote Desktop Protocol | 95.96% | 99 | 88.89% | 99 |
| T1204.002 | Malicious File | 93.75% | 96 | 82.29% | 96 |
| T1083 | File and Directory Discovery | 78.49% | 93 | 92.47% | 93 |
| T1562.001 | Disable or Modify Tools | 83.87% | 93 | 92.47% | 93 |
| T1070.004 | File Deletion | 76.67% | 90 | 82.22% | 90 |
| T1566.001 | Spearphishing Attachment | 88.89% | 90 | 93.33% | 90 |
| T1041 | Exfiltration Over C2 Channel | 38.75% | 80 | 70.00% | 80 |
| T1047 | Windows Management Instrumentation | 90.91% | 77 | 89.61% | 77 |
| T1057 | Process Discovery | 89.61% | 77 | 85.71% | 77 |
| T1574.002 | DLL Side-Loading | 94.74% | 76 | 90.79% | 76 |
| T1036.005 | Match Legitimate Name or Location | 81.69% | 71 | 63.38% | 71 |
| T1547.001 | Registry Run Keys / Startup Folder | 89.23% | 65 | 92.31% | 65 |
| T1056.001 | Keylogging | 90.32% | 62 | 93.55% | 62 |
| T1005 | Data from Local System | 47.54% | 61 | 75.41% | 61 |
| T1190 | Exploit Public-Facing Application | 96.72% | 61 | 85.25% | 61 |
| T1218.011 | Rundll32 | 90.00% | 60 | 73.33% | 60 |
| T1573.001 | Symmetric Cryptography | 66.67% | 60 | 88.33% | 60 |
| T1570 | Lateral Tool Transfer | 54.24% | 59 | 81.36% | 59 |
| T1095 | Non-Application Layer Protocol | 89.47% | 57 | 91.23% | 57 |
| T1219 | Remote Access Software | 91.07% | 56 | 73.21% | 56 |
| T1543.003 | Windows Service | 88.68% | 53 | 86.79% | 53 |
| T1016 | System Network Configuration Discovery | 88.46% | 52 | 84.62% | 52 |
| T1110 | Brute Force | 71.15% | 52 | 86.54% | 52 |
| T1113 | Screen Capture | 75.51% | 49 | 87.76% | 49 |
| T1033 | System Owner/User Discovery | 80.85% | 47 | 87.23% | 47 |
| T1518.001 | Security Software Discovery | 83.78% | 37 | 70.27% | 37 |
| T1074.001 | Local Data Staging | 65.38% | 26 | 88.46% | 26 |
| T1548.002 | Bypass User Account Control | 92.31% | 26 | 92.31% | 26 |
| T1012 | Query Registry | 58.33% | 24 | 83.33% | 24 |
| T1484.001 | Group Policy Modification | 95.83% | 24 | 91.67% | 24 |
| T1552.001 | Credentials In Files | 86.36% | 22 | 90.91% | 22 |
| T1569.002 | Service Execution | 77.27% | 22 | 81.82% | 22 |
| T1210 | Exploitation of Remote Services | 88.24% | 17 | 88.24% | 17 |
| T1564.001 | Hidden Files and Directories | 87.50% | 16 | 100.00% | 16 |
| T1068 | Exploitation for Privilege Escalation | 90.00% | 10 | 90.00% | 10 |
| T1072 | Software Deployment Tools | 60.00% | 10 | 60.00% | 10 |
| T1557.001 | LLMNR/NBT-NS Poisoning and SMB Relay | 85.71% | 7 | 100.00% | 7 |

## Best 5 by Nomic Recall@10

| Technique | Name | R@10 | Support |
| :--- | :--- | ---: | ---: |
| T1190 | Exploit Public-Facing Application | 96.72% | 61 |
| T1021.001 | Remote Desktop Protocol | 95.96% | 99 |
| T1484.001 | Group Policy Modification | 95.83% | 24 |
| T1574.002 | DLL Side-Loading | 94.74% | 76 |
| T1003.001 | LSASS Memory | 94.64% | 112 |

## Worst 5 by Nomic Recall@10

| Technique | Name | R@10 | Support |
| :--- | :--- | ---: | ---: |
| T1041 | Exfiltration Over C2 Channel | 38.75% | 80 |
| T1005 | Data from Local System | 47.54% | 61 |
| T1570 | Lateral Tool Transfer | 54.24% | 59 |
| T1012 | Query Registry | 58.33% | 24 |
| T1072 | Software Deployment Tools | 60.00% | 10 |

## Best 5 by ATTACK-BERT Recall@10

| Technique | Name | R@10 | Support |
| :--- | :--- | ---: | ---: |
| T1557.001 | LLMNR/NBT-NS Poisoning and SMB Relay | 100.00% | 7 |
| T1564.001 | Hidden Files and Directories | 100.00% | 16 |
| T1140 | Deobfuscate/Decode Files or Information | 93.99% | 466 |
| T1056.001 | Keylogging | 93.55% | 62 |
| T1566.001 | Spearphishing Attachment | 93.33% | 90 |

## Worst 5 by ATTACK-BERT Recall@10

| Technique | Name | R@10 | Support |
| :--- | :--- | ---: | ---: |
| T1059.003 | Windows Command Shell | 53.35% | 358 |
| T1106 | Native API | 54.11% | 207 |
| T1072 | Software Deployment Tools | 60.00% | 10 |
| T1036.005 | Match Legitimate Name or Location | 63.38% | 71 |
| T1105 | Ingress Tool Transfer | 65.52% | 261 |
