#include "PHY/INIT/nr_phy_init.h"
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PHY/defs_gNB.h"
#include "PHY/defs_RU.h"
#include "radio/COMMON/common_lib.h"
typedef struct {
 uint64_t magic;uint32_t version,header_size;int frame,slot,symbol,kind;
 int chest_time,chest_freq,dmrs_antennas,symbols_per_task,cell_id,rnti,max_ch;uint32_t nvar;
 double gain,gain_offset;
 NR_DL_FRAME_PARMS fp;nfapi_nr_pusch_pdu_t pdu;NR_gNB_PUSCH pusch;
 PHY_MEASUREMENTS_gNB meas;ulsch_measurements_gNB ulmeas;delay_t delay;
} tb_header_t;
typedef struct {tb_header_t *h;PHY_VARS_gNB *g;int16_t *scramble;uint32_t G;size_t rx_count,ch_count,comp_count;int16_t valid[14];} tb_context_t;
static void *tb_alloc(size_t n){void*p=NULL;AssertFatal(posix_memalign(&p,64,n?n:64)==0,"alloc");memset(p,0,n);return p;}
static unsigned tb_bits(const nfapi_nr_pusch_pdu_t*p){int dmrs=0;for(int s=p->start_symbol_index;s<p->start_symbol_index+p->nr_of_symbols;s++)dmrs+=(p->ul_dmrs_symb_pos>>s)&1;return nr_get_G(p->rb_size,p->nr_of_symbols,(p->dmrs_config_type==pusch_dmrs_type1?6:4)*p->num_dmrs_cdm_grps_no_data,dmrs,0,p->qam_mod_order,p->nrOfLayers);}
static uint32_t tb_rng(uint32_t*s){uint32_t x=*s;x^=x<<13;x^=x>>17;x^=x<<5;return *s=x;}
static tb_context_t *tb_random(uint32_t seed,int rb){
 tb_context_t*c=tb_alloc(sizeof(*c));c->h=tb_alloc(sizeof(*c->h));tb_header_t*h=c->h;
 h->kind=TB_KIND;h->slot=0;h->symbol=0;h->rnti=0x1234;h->gain=1;h->dmrs_antennas=1;h->symbols_per_task=1;
 h->fp.N_RB_UL=106;h->fp.N_RB_DL=106;h->fp.ofdm_symbol_size=2048;h->fp.first_carrier_offset=1412;
 h->fp.nb_antennas_rx=1;h->fp.nb_antennas_tx=1;h->fp.symbols_per_slot=14;h->fp.slots_per_frame=20;h->fp.slots_per_subframe=2;
 h->fp.numerology_index=1;h->fp.nb_prefix_samples=144;h->fp.nb_prefix_samples0=176;
 init_delay_table(2048,MAX_DELAY_COMP,NR_MAX_OFDM_SYMBOL_SIZE,h->fp.delay_table);
 init_delay_table(128,MAX_DELAY_COMP,128,h->fp.delay_table128);
 h->pdu.rb_size=rb;h->pdu.bwp_size=106;h->pdu.nrOfLayers=1;h->pdu.qam_mod_order=4;h->pdu.nr_of_symbols=13;
 h->pdu.ul_dmrs_symb_pos=1057;h->pdu.dmrs_config_type=pusch_dmrs_type1;h->pdu.num_dmrs_cdm_grps_no_data=1;
 h->pdu.dmrs_ports=1;h->pdu.transform_precoding=transformPrecoder_disabled;h->pdu.subcarrier_spacing=1;h->pdu.rnti=h->rnti;
 h->pusch.log2_maxh=12;h->max_ch=1048576;h->nvar=1024;
 c->g=tb_alloc(sizeof(*c->g));PHY_VARS_gNB*g=c->g;g->frame_parms=h->fp;g->chest_time=h->chest_time;g->chest_freq=h->chest_freq;g->gNB_config.cell_config.phy_cell_id.value=h->cell_id;g->dmrs_num_antennas_per_thread=h->dmrs_antennas;g->num_pusch_symbols_per_thread=h->symbols_per_task;g->pusch_vars=tb_alloc(sizeof(*g->pusch_vars));g->ulsch=tb_alloc(sizeof(*g->ulsch));g->ulsch[0].harq_process=tb_alloc(sizeof(*g->ulsch[0].harq_process));g->ulsch[0].harq_process->ulsch_pdu=h->pdu;g->ulsch[0].rnti=h->rnti;
 g->RU_list[0]=tb_alloc(sizeof(RU_t));g->RU_list[0]->rfdevice.openair0_cfg=tb_alloc(sizeof(openair0_config_t));g->RU_list[0]->rfdevice.openair0_cfg->rx_gain[0]=h->gain;g->RU_list[0]->rfdevice.openair0_cfg->rx_gain_offset[0]=h->gain_offset;
 c->rx_count=RU_RX_SLOT_DEPTH*h->fp.symbols_per_slot*h->fp.ofdm_symbol_size;c->ch_count=h->fp.symbols_per_slot*h->fp.ofdm_symbol_size;c->comp_count=h->fp.symbols_per_slot*ceil_mod(h->fp.N_RB_UL*12,16);c->G=tb_bits(&h->pdu);
 g->common_vars.rxdataF=tb_alloc(sizeof(c16_t**));g->common_vars.rxdataF[0]=tb_alloc(sizeof(c16_t*));g->common_vars.rxdataF[0][0]=tb_alloc(c->rx_count*sizeof(c16_t));
 NR_gNB_PUSCH*v=g->pusch_vars;*v=h->pusch;v->ul_ch_estimates=tb_alloc(sizeof(int32_t*));v->ul_ch_estimates[0]=tb_alloc(c->ch_count*sizeof(c16_t));v->rxdataF_comp=tb_alloc(sizeof(int32_t*));v->rxdataF_comp[0]=tb_alloc(c->comp_count*sizeof(c16_t));v->llr=tb_alloc((c->G+128)*sizeof(int16_t));v->ul_valid_re_per_slot=tb_alloc(14*sizeof(int16_t));v->ptrs_phase_per_slot=tb_alloc(sizeof(int32_t*));v->ptrs_phase_per_slot[0]=tb_alloc(14*sizeof(int32_t));
 g->measurements=h->meas;g->measurements.n0_subband_power=allocateFourDimArray(sizeof(unsigned),1,h->fp.N_RB_UL,0,0);g->measurements.n0_subband_power_dB=allocateFourDimArray(sizeof(unsigned),1,h->fp.N_RB_UL,0,0);c->scramble=tb_alloc(c->G*sizeof(int16_t));
 uint32_t rng=seed?seed:1;
 int16_t*rx=(int16_t*)g->common_vars.rxdataF[0][0];for(size_t i=0;i<2*c->rx_count;i++)rx[i]=(int16_t)((tb_rng(&rng)&4095)-2048);
 int16_t*ch=(int16_t*)v->ul_ch_estimates[0];for(size_t i=0;i<2*c->ch_count;i++)ch[i]=(int16_t)((tb_rng(&rng)&4095)-2048);
 for(int i=0;i<14;i++)v->ul_valid_re_per_slot[i]=i<13?rb*(((1057>>i)&1)?6:12):0;
 memcpy(c->valid,v->ul_valid_re_per_slot,sizeof(c->valid));
 for(unsigned i=0;i<c->G;i++)c->scramble[i]=(tb_rng(&rng)&1)?-1:1;
 for(int i=0;i<106;i++){((unsigned*)g->measurements.n0_subband_power->data)[i]=1024+(tb_rng(&rng)&1023);}
 g->ulsch[0].ulsch_measurements=h->ulmeas;g->ulsch[0].delay=h->delay;return c;
}
static void tb_restore(tb_context_t*c,const tb_context_t*base){
 PHY_VARS_gNB*g=c->g,*b=base->g;NR_gNB_PUSCH*v=g->pusch_vars,*s=b->pusch_vars;int32_t**ch=v->ul_ch_estimates,**comp=v->rxdataF_comp,**phase=v->ptrs_phase_per_slot;int16_t*llr=v->llr,*valid=v->ul_valid_re_per_slot;*v=*s;v->ul_ch_estimates=ch;v->rxdataF_comp=comp;v->ptrs_phase_per_slot=phase;v->llr=llr;v->ul_valid_re_per_slot=valid;
 memcpy(ch[0],s->ul_ch_estimates[0],c->ch_count*sizeof(c16_t));memcpy(comp[0],s->rxdataF_comp[0],c->comp_count*sizeof(c16_t));memcpy(llr,s->llr,c->G*sizeof(int16_t));memcpy(valid,s->ul_valid_re_per_slot,14*sizeof(int16_t));
 fourDimArray_t*n0=g->measurements.n0_subband_power,*n0db=g->measurements.n0_subband_power_dB;g->measurements=b->measurements;g->measurements.n0_subband_power=n0;g->measurements.n0_subband_power_dB=n0db;memcpy(n0->data,b->measurements.n0_subband_power->data,c->h->fp.N_RB_UL*sizeof(unsigned));memcpy(n0db->data,b->measurements.n0_subband_power_dB->data,c->h->fp.N_RB_UL*sizeof(unsigned));g->ulsch[0].ulsch_measurements=b->ulsch[0].ulsch_measurements;g->ulsch[0].delay=b->ulsch[0].delay;c->h->max_ch=base->h->max_ch;c->h->nvar=base->h->nvar;
}

