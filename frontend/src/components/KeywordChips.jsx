import { motion } from "framer-motion";

function KeywordChips({ keywords = [] }) {
  if (!keywords.length) {
    return <p className="muted-copy">No TF-IDF keyword explanation was available for this sample.</p>;
  }

  return (
    <motion.div
      className="keyword-grid"
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: {
          transition: {
            staggerChildren: 0.08,
          },
        },
      }}
    >
      {keywords.map((keyword, index) => (
        <motion.div
          key={`${keyword.feature}-${keyword.contribution}`}
          className="keyword-chip"
          variants={{
            hidden: { opacity: 0, y: 12 },
            visible: { opacity: 1, y: 0 },
          }}
        >
          <small>#{index + 1}</small>
          <span>{keyword.feature}</span>
          <strong>{Number(keyword.contribution).toFixed(4)}</strong>
        </motion.div>
      ))}
    </motion.div>
  );
}

export default KeywordChips;
