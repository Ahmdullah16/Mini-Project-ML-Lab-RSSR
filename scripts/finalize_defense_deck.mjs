/** Finalize a native deck with existing embedded chart workbooks and exact caches. */
import fs from 'node:fs/promises';import path from 'node:path';import {pathToFileURL,fileURLToPath} from 'node:url';
process.env.RUNTIME_NODE_MODULES ??= await fs.realpath(path.join(path.dirname(fileURLToPath(import.meta.url)),'node_modules'));
const [workspaceArg,candidateArg,name='RSSR_Defense_Presentation.pptx',receipt='presentation-validation.json']=process.argv.slice(2);
const workspaceDir=path.resolve(workspaceArg),candidatePath=path.resolve(candidateArg),skill=process.env.RSSR_PRESENTATION_SKILL;
if(path.basename(name)!==name||path.basename(receipt)!==receipt)throw Error('Output names must be basenames');
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({workspaceDir,candidatePath,finalPath:path.join(workspaceDir,'output',name),pythonExecutable:process.env.RSSR_ARTIFACT_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','4'],explicitTotalSlideCount:30,requiredNativeTableOwnerSlides:[4],requiredNativeChartOwnerSlides:[12,13],requiredEmbeddedWorkbookChartOwnerSlides:[12,13],materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(workspaceDir,receipt)});
console.log('Native deck finalized successfully');
