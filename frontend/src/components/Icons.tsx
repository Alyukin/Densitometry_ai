import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function base({ size = 16, ...rest }: P) {
  return {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
    ...rest,
  };
}

export const IconUpload = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 16V4" />
    <path d="m6 10 6-6 6 6" />
    <path d="M4 20h16" />
  </svg>
);

export const IconPlay = (p: P) => (
  <svg {...base(p)}>
    <path d="M7 4.5v15l12-7.5z" />
  </svg>
);

export const IconRefresh = (p: P) => (
  <svg {...base(p)}>
    <path d="M20 11a8 8 0 1 0-2.3 5.7" />
    <path d="M20 4v7h-7" />
  </svg>
);

export const IconDownload = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 4v12" />
    <path d="m6 10 6 6 6-6" />
    <path d="M4 20h16" />
  </svg>
);

export const IconTrash = (p: P) => (
  <svg {...base(p)}>
    <path d="M4 7h16" />
    <path d="M10 11v6M14 11v6" />
    <path d="M6 7l1 13h10l1-13" />
    <path d="M9 7V4h6v3" />
  </svg>
);

export const IconClose = (p: P) => (
  <svg {...base(p)}>
    <path d="M6 6l12 12M18 6 6 18" />
  </svg>
);

export const IconCheck = (p: P) => (
  <svg {...base(p)}>
    <path d="m5 12 5 5 9-10" />
  </svg>
);

export const IconAlert = (p: P) => (
  <svg {...base(p)}>
    <path d="M12 3 2 20h20z" />
    <path d="M12 10v4M12 17.5v.5" />
  </svg>
);

export const IconFile = (p: P) => (
  <svg {...base(p)}>
    <path d="M14 3H6v18h12V7z" />
    <path d="M14 3v4h4" />
  </svg>
);

export const IconFolder = (p: P) => (
  <svg {...base(p)}>
    <path d="M3 6h6l2 2h10v11H3z" />
  </svg>
);

export const IconExternal = (p: P) => (
  <svg {...base(p)}>
    <path d="M14 4h6v6" />
    <path d="M20 4 10 14" />
    <path d="M18 14v6H4V6h6" />
  </svg>
);

export const IconChevron = (p: P) => (
  <svg {...base(p)}>
    <path d="m9 6 6 6-6 6" />
  </svg>
);

export const IconInfo = (p: P) => (
  <svg {...base(p)}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8v.5" />
  </svg>
);

export const Logo = ({ size = 32 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <rect width="32" height="32" rx="8" fill="var(--accent)" />
    <g fill="#fff">
      <rect x="12" y="5" width="8" height="5" rx="1.5" />
      <rect x="12" y="11.5" width="8" height="5" rx="1.5" opacity=".85" />
      <rect x="12" y="18" width="8" height="5" rx="1.5" opacity=".7" />
      <rect x="12" y="24.5" width="8" height="3" rx="1.5" opacity=".55" />
    </g>
  </svg>
);
