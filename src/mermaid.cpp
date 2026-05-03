#include "mermaid.h"

#include <sstream>

namespace sllmrf::mermaid {

std::string compact_node_id(std::string_view name) {
    std::string id = "compact_";
    for (char character : name) {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9')) {
            id += character;
        } else {
            id += '_';
        }
    }
    return id;
}

std::string escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char character : value) {
        switch (character) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
            case '\r':
                escaped += ' ';
                break;
            case '<':
                escaped += "&lt;";
                break;
            case '>':
                escaped += "&gt;";
                break;
            case '&':
                escaped += "&amp;";
                break;
            default:
                escaped += character;
                break;
        }
    }
    return escaped;
}

std::string node_label(std::string_view title, std::string_view detail) {
    return escape(title) + "<br/>" + escape(detail);
}

void write_class_defs(std::ostringstream &stream) {
    stream << "    classDef io fill:#eef6ff,stroke:#4c78a8,color:#111827\n";
    stream << "    classDef file fill:#f8fafc,stroke:#475569,color:#111827\n";
    stream << "    classDef metadata fill:#fefce8,stroke:#a16207,color:#111827\n";
    stream << "    classDef embedding fill:#ecfdf5,stroke:#2f855a,color:#111827\n";
    stream << "    classDef norm fill:#fff7ed,stroke:#c05621,color:#111827\n";
    stream << "    classDef linear fill:#f5f3ff,stroke:#6b46c1,color:#111827\n";
    stream << "    classDef attention fill:#fef2f2,stroke:#c53030,color:#111827\n";
    stream << "    classDef residual fill:#f8fafc,stroke:#475569,color:#111827\n";
    stream << "    classDef activation fill:#f0fdfa,stroke:#0f766e,color:#111827\n";
    stream << "    classDef tensor fill:#eff6ff,stroke:#2563eb,color:#111827\n";
    stream << "    classDef runtime fill:#fdf2f8,stroke:#be185d,color:#111827\n";
    stream << "    classDef group fill:#ffffff,stroke:#64748b,color:#111827\n";
}

}  // namespace sllmrf::mermaid
