import { useEffect, useState } from "react";

export const useNarrow = (px = 576) => {
  const [narrow, setNarrow] = useState(
    typeof window !== "undefined" && window.innerWidth < px,
  );
  useEffect(() => {
    const h = () => setNarrow(window.innerWidth < px);
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [px]);
  return narrow;
};
