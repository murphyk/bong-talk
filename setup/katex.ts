import { defineKatexSetup } from '@slidev/types'

export default defineKatexSetup(() => {
  return {
    maxExpand: 2000,
    macros: {
	"\\test": "hello, world",
	"\\vA": "{\\bf A}",
	"\\data": "{\\cal D}",
}
  }
})