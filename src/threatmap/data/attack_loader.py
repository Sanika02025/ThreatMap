"""
MITRE ATT&CK Knowledge Base loader, parser, and manager.
Supports STIX 2.0/2.1 Enterprise ATT&CK bundles and normalized JSON cache.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from threatmap.logging_config import setup_logger
from threatmap.schemas import AttackTechnique

logger = setup_logger("threatmap.attack_loader")

# Official mapping from kill-chain phase names to standardized Tactic Display Names
TACTIC_MAP = {
    "initial-access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "collection": "Collection",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
    "reconnaissance": "Reconnaissance",
    "resource-development": "Resource Development",
}

# Verified miniature fixture for zero-download testing and offline unit tests
MINIATURE_ATTACK_FIXTURE: List[Dict[str, Any]] = [
    {
        "technique_id": "T1566.001",
        "technique_name": "Phishing: Spearphishing Attachment",
        "description": "Adversaries may send spearphishing emails with a malicious attachment in an attempt to gain access to victim systems.",
        "tactic_names": ["Initial Access"],
        "tactic_ids": ["TA0001"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Network intrusion detection systems and email gateway filters can detect suspicious attachments.",
        "data_sources": ["Application Log", "File", "Network Traffic"],
        "source_url": "https://attack.mitre.org/techniques/T1566/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1566",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1059.001",
        "technique_name": "Command and Scripting Interpreter: PowerShell",
        "description": "Adversaries may abuse PowerShell commands and scripts for execution. PowerShell is a powerful interactive command-line interface and scripting language.",
        "tactic_names": ["Execution"],
        "tactic_ids": ["TA0002"],
        "platforms": ["Windows"],
        "detection": "Enable Script Block Logging (Event ID 4104) and transcription logging for PowerShell.",
        "data_sources": ["Command", "Process", "Script"],
        "source_url": "https://attack.mitre.org/techniques/T1059/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1059",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1053.005",
        "technique_name": "Scheduled Task/Job: Scheduled Task",
        "description": "Adversaries may abuse the Windows Task Scheduler to schedule programs or scripts for initial or recurring execution to maintain persistence.",
        "tactic_names": ["Execution", "Persistence", "Privilege Escalation"],
        "tactic_ids": ["TA0002", "TA0003", "TA0004"],
        "platforms": ["Windows"],
        "detection": "Monitor process creation of schtasks.exe and Windows Event ID 4698 (A scheduled task was created).",
        "data_sources": ["Command", "Process", "Scheduled Task"],
        "source_url": "https://attack.mitre.org/techniques/T1053/005/",
        "is_subtechnique": True,
        "subtechnique_of": "T1053",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1003.001",
        "technique_name": "OS Credential Dumping: LSASS Memory",
        "description": "Adversaries may attempt to access credential material stored in the process memory of the Local Security Authority Subsystem Service (LSASS).",
        "tactic_names": ["Credential Access"],
        "tactic_ids": ["TA0006"],
        "platforms": ["Windows"],
        "detection": "Monitor access to the lsass.exe process by unauthorized processes (Sysmon Event ID 10).",
        "data_sources": ["Process", "Command"],
        "source_url": "https://attack.mitre.org/techniques/T1003/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1003",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1033",
        "technique_name": "System Owner/User Discovery",
        "description": "Adversaries may attempt to identify the primary user, currently logged-in user, set of users that commonly use a system, or whether a user is actively at the keyboard using utilities like whoami.",
        "tactic_names": ["Discovery"],
        "tactic_ids": ["TA0007"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Monitor command executions such as whoami, id, or quser.",
        "data_sources": ["Command", "Process"],
        "source_url": "https://attack.mitre.org/techniques/T1033/",
        "is_subtechnique": False,
        "subtechnique_of": None,
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1087.001",
        "technique_name": "Account Discovery: Local Account",
        "description": "Adversaries may attempt to get a listing of local system accounts using commands like net user.",
        "tactic_names": ["Discovery"],
        "tactic_ids": ["TA0007"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Monitor commands such as net user, dscl, or reading /etc/passwd.",
        "data_sources": ["Command", "Process"],
        "source_url": "https://attack.mitre.org/techniques/T1087/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1087",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1021.001",
        "technique_name": "Remote Services: Remote Desktop Protocol",
        "description": "Adversaries may use Valid Accounts to log into a computer using the Remote Desktop Protocol (RDP) to expand access.",
        "tactic_names": ["Lateral Movement"],
        "tactic_ids": ["TA0008"],
        "platforms": ["Windows"],
        "detection": "Monitor network connections to port 3389 and Windows Event ID 4624 (Logon Type 10).",
        "data_sources": ["Network Traffic", "Logon Session"],
        "source_url": "https://attack.mitre.org/techniques/T1021/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1021",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1070.001",
        "technique_name": "Indicator Removal: Clear Windows Event Logs",
        "description": "Adversaries may clear Windows Event Logs using wevtutil or PowerShell to hide the activity of an intrusion.",
        "tactic_names": ["Defense Evasion"],
        "tactic_ids": ["TA0005"],
        "platforms": ["Windows"],
        "detection": "Monitor for Windows Event ID 1102 (The audit log was cleared) and Event ID 104 (System log cleared).",
        "data_sources": ["Command", "Windows Event Log"],
        "source_url": "https://attack.mitre.org/techniques/T1070/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1070",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1560.001",
        "technique_name": "Archive Collected Data: Archive via Utility",
        "description": "Adversaries may use third-party or built-in compression utilities such as 7-Zip, WinRAR, or tar to compress data before exfiltration.",
        "tactic_names": ["Collection"],
        "tactic_ids": ["TA0009"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Monitor command execution of rar.exe, 7z.exe, zip, or tar commands.",
        "data_sources": ["Command", "Process", "File"],
        "source_url": "https://attack.mitre.org/techniques/T1560/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1560",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1041",
        "technique_name": "Exfiltration Over C2 Channel",
        "description": "Adversaries may steal data by transferring it through the existing command and control channel rather than initiating a new connection.",
        "tactic_names": ["Exfiltration"],
        "tactic_ids": ["TA0010"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Monitor egress network bandwidth and anomalous spikes during communication with known C2 servers.",
        "data_sources": ["Network Traffic"],
        "source_url": "https://attack.mitre.org/techniques/T1041/",
        "is_subtechnique": False,
        "subtechnique_of": None,
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1105",
        "technique_name": "Ingress Tool Transfer",
        "description": "Adversaries may transfer tools or other files from an external system into a compromised environment.",
        "tactic_names": ["Command and Control"],
        "tactic_ids": ["TA0011"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Monitor file creation and network activity initiated by command-line interpreters or scripts.",
        "data_sources": ["File", "Network Traffic"],
        "source_url": "https://attack.mitre.org/techniques/T1105/",
        "is_subtechnique": False,
        "subtechnique_of": None,
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1218.005",
        "technique_name": "System Binary Proxy Execution: Mshta",
        "description": "Adversaries may abuse mshta.exe to proxy execution of malicious .hta files and embedded scripts.",
        "tactic_names": ["Defense Evasion"],
        "tactic_ids": ["TA0005"],
        "platforms": ["Windows"],
        "detection": "Monitor process arguments of mshta.exe pointing to remote URLs or suspicious local scripts.",
        "data_sources": ["Command", "Process"],
        "source_url": "https://attack.mitre.org/techniques/T1218/005/",
        "is_subtechnique": True,
        "subtechnique_of": "T1218",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1055.012",
        "technique_name": "Process Injection: Process Hollowing",
        "description": "Adversaries may inject malicious code into suspended and hollowed processes like svchost.exe to evade process-based defenses.",
        "tactic_names": ["Defense Evasion", "Privilege Escalation"],
        "tactic_ids": ["TA0005", "TA0004"],
        "platforms": ["Windows"],
        "detection": "Monitor API calls associated with unmapping and repopulating process memory (NtUnmapViewOfSection).",
        "data_sources": ["Process"],
        "source_url": "https://attack.mitre.org/techniques/T1055/012/",
        "is_subtechnique": True,
        "subtechnique_of": "T1055",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1547.001",
        "technique_name": "Boot or Logon Autostart Execution: Registry Run Keys / Startup Folder",
        "description": "Adversaries may achieve persistence by adding a program to a registry Run key or the Windows Startup folder.",
        "tactic_names": ["Persistence", "Privilege Escalation"],
        "tactic_ids": ["TA0003", "TA0004"],
        "platforms": ["Windows"],
        "detection": "Monitor modifications to registry keys under CurrentVersion\\Run.",
        "data_sources": ["Windows Registry"],
        "source_url": "https://attack.mitre.org/techniques/T1547/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1547",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
    {
        "technique_id": "T1071.001",
        "technique_name": "Application Layer Protocol: Web Protocols",
        "description": "Adversaries may communicate using application layer protocols (HTTP/HTTPS) to avoid detection through firewalls and proxies.",
        "tactic_names": ["Command and Control"],
        "tactic_ids": ["TA0011"],
        "platforms": ["Windows", "macOS", "Linux"],
        "detection": "Analyze HTTP/HTTPS traffic patterns, TLS certificates, and domain reputation.",
        "data_sources": ["Network Traffic"],
        "source_url": "https://attack.mitre.org/techniques/T1071/001/",
        "is_subtechnique": True,
        "subtechnique_of": "T1071",
        "revoked": False,
        "deprecated": False,
        "version": "14.1",
    },
]


class AttackKnowledgeBase:
    """In-memory index and lookup service for MITRE ATT&CK techniques."""

    def __init__(self, techniques: Optional[List[AttackTechnique]] = None, source_info: str = "Uninitialized"):
        self.techniques: Dict[str, AttackTechnique] = {}
        self.source_info = source_info
        if techniques:
            for t in techniques:
                self.techniques[t.technique_id] = t

    def __len__(self) -> int:
        return len(self.techniques)

    def get_technique(self, technique_id: str) -> Optional[AttackTechnique]:
        """Lookup technique by ID."""
        return self.techniques.get(technique_id.strip())

    def get_all_active_techniques(self, include_deprecated: bool = False) -> List[AttackTechnique]:
        """Returns list of active (non-revoked, non-deprecated) techniques."""
        results = []
        for t in self.techniques.values():
            if t.revoked:
                continue
            if not include_deprecated and t.deprecated:
                continue
            results.append(t)
        return results

    def get_technique_ids(self, include_deprecated: bool = False) -> Set[str]:
        """Returns set of all valid technique IDs."""
        return {t.technique_id for t in self.get_all_active_techniques(include_deprecated)}

    def search_by_name(self, name_query: str) -> List[AttackTechnique]:
        """Case-insensitive search by technique name."""
        q = name_query.lower()
        return [t for t in self.techniques.values() if q in t.technique_name.lower()]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serializes knowledge base to list of dictionaries."""
        return [t.model_dump() for t in self.techniques.values()]

    def save_to_json(self, file_path: str | Path) -> None:
        """Saves normalized knowledge base to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_info": self.source_info,
            "count": len(self.techniques),
            "techniques": self.to_dict_list(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.techniques)} ATT&CK techniques to {path}")

    @classmethod
    def load_from_json(cls, file_path: str | Path) -> "AttackKnowledgeBase":
        """Loads normalized knowledge base from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Processed ATT&CK file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_techniques = data.get("techniques", [])
        techniques = [AttackTechnique(**t) for t in raw_techniques]
        source_info = data.get("source_info", f"Loaded from {path.name}")
        return cls(techniques=techniques, source_info=source_info)


def parse_attack_stix_bundle(stix_data: Dict[str, Any]) -> AttackKnowledgeBase:
    """
    Parses official MITRE Enterprise STIX 2.0/2.1 JSON bundle.
    Resolves sub-technique relationships and kill-chain tactics.
    """
    objects = stix_data.get("objects", [])
    if not objects:
        raise ValueError("Invalid STIX bundle: 'objects' list is empty or missing")

    attack_patterns: Dict[str, Dict[str, Any]] = {}
    subtechnique_relationships: Dict[str, str] = {}  # source_ref (sub) -> target_ref (parent)

    # First pass: collect attack-patterns and relationships
    for obj in objects:
        obj_type = obj.get("type")
        if obj_type == "attack-pattern":
            attack_patterns[obj["id"]] = obj
        elif obj_type == "relationship" and obj.get("relationship_type") == "subtechnique-of":
            subtechnique_relationships[obj.get("source_ref")] = obj.get("target_ref")

    # Map STIX object ID to official ATT&CK technique ID (e.g. attack-pattern--... -> T1059.001)
    stix_id_to_attack_id: Dict[str, str] = {}
    for stix_id, obj in attack_patterns.items():
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and "external_id" in ref:
                stix_id_to_attack_id[stix_id] = ref["external_id"]
                break

    parsed_techniques: List[AttackTechnique] = []

    for stix_id, obj in attack_patterns.items():
        attack_id = stix_id_to_attack_id.get(stix_id)
        if not attack_id:
            continue

        name = obj.get("name", "Unknown Technique")
        description = obj.get("description", "")
        revoked = bool(obj.get("revoked", False))
        deprecated = bool(obj.get("x_mitre_deprecated", False))
        is_subtechnique = bool(obj.get("x_mitre_is_subtechnique", False))
        platforms = obj.get("x_mitre_platforms", [])
        detection = obj.get("x_mitre_detection")
        data_sources = obj.get("x_mitre_data_sources", [])
        version = obj.get("x_mitre_version")

        # Extract tactics from kill chain phases
        tactic_names: List[str] = []
        for phase in obj.get("kill_chain_phases", []):
            phase_name = phase.get("phase_name")
            if phase_name and phase_name in TACTIC_MAP:
                tactic_names.append(TACTIC_MAP[phase_name])
            elif phase_name:
                tactic_names.append(phase_name.replace("-", " ").title())

        # Resolve parent if subtechnique
        subtechnique_of = None
        if is_subtechnique and stix_id in subtechnique_relationships:
            parent_stix_id = subtechnique_relationships[stix_id]
            subtechnique_of = stix_id_to_attack_id.get(parent_stix_id)

        source_url = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                source_url = ref.get("url")
                break

        tech = AttackTechnique(
            technique_id=attack_id,
            technique_name=name,
            description=description,
            tactic_names=list(set(tactic_names)),
            tactic_ids=[],
            platforms=platforms,
            detection=detection,
            data_sources=data_sources,
            source_url=source_url,
            is_subtechnique=is_subtechnique,
            subtechnique_of=subtechnique_of,
            revoked=revoked,
            deprecated=deprecated,
            version=version,
        )
        parsed_techniques.append(tech)

    spec_version = stix_data.get("spec_version", "2.1")
    return AttackKnowledgeBase(
        techniques=parsed_techniques,
        source_info=f"Official MITRE ATT&CK STIX Bundle (Parsed {len(parsed_techniques)} techniques, spec {spec_version})",
    )


def load_attack_data(
    processed_path: Optional[str | Path] = None,
    raw_path: Optional[str | Path] = None,
    fallback_to_fixture: bool = True,
) -> AttackKnowledgeBase:
    """
    Primary data loader:
    1. Tries loading processed normalized JSON.
    2. If missing, tries loading raw STIX JSON and parsing it.
    3. If both missing and fallback allowed, uses verified miniature fixture with clear logging.
    """
    if processed_path:
        p_path = Path(processed_path)
        if p_path.exists():
            logger.info(f"Loading processed ATT&CK data from {p_path}")
            return AttackKnowledgeBase.load_from_json(p_path)

    if raw_path:
        r_path = Path(raw_path)
        if r_path.exists():
            logger.info(f"Parsing raw ATT&CK STIX data from {r_path}")
            with open(r_path, "r", encoding="utf-8") as f:
                stix_data = json.load(f)
            kb = parse_attack_stix_bundle(stix_data)
            if processed_path:
                kb.save_to_json(processed_path)
            return kb

    if fallback_to_fixture:
        logger.warning(
            "ATT&CK data files not found. Loading verified miniature fixture. "
            "Run scripts/download_attack_data.py to download the complete official Enterprise ATT&CK dataset."
        )
        techniques = [AttackTechnique(**t) for t in MINIATURE_ATTACK_FIXTURE]
        kb = AttackKnowledgeBase(
            techniques=techniques,
            source_info="Verified Miniature ATT&CK Fixture (15 core techniques for offline testing)",
        )
        if processed_path:
            kb.save_to_json(processed_path)
        return kb

    raise FileNotFoundError("No ATT&CK data files found and fallback is disabled.")
